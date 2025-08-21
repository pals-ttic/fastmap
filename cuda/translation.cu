#include "kernels.h"
#include <cassert>
#include <cmath>

// define aliases for shared memory buffers
#define O1 sharedBuffer3x1a
#define O2 sharedBuffer3x1b
#define O12 sharedBuffer3x1a // Overwrite O1
#define O12_NORM sharedBuffer1x1a
#define O12_NORMALIZED sharedBuffer3x1a // Overwrite O12
#define O12_GT sharedBuffer3x1b         // Overwrite O2
#define LOSS sharedBuffer1x1b
#define d_O12_NORMALIZED sharedBuffer3x1c
#define d_O12 sharedBuffer3x1b // Overwrite O12_GT
#define d_O2 sharedBuffer3x1b  // Overwrite d_O12
#define d_O1 sharedBuffer3x1b  // Overwrite d_O2

constexpr int WARP_SIZE = 32;
constexpr int BATCH_SIZE = 8; // Adjust as needed

// Function to copy contiguous memory from source to target
__device__ __forceinline__ void
copyContiguousMemory(const char *__restrict__ sourcePtr,
                     char *__restrict__ targetPtr, int numBytes) {
  const int threadIdxInBlock = threadIdx.x;
  const int numThreadsInBlock = blockDim.x;
  for (int i = threadIdxInBlock; i < numBytes; i += numThreadsInBlock) {
    targetPtr[i] = sourcePtr[i];
  }
}

template <typename T>
__global__ void
translationKernel(const T *__restrict__ o1GlobalPtr,
                  const T *__restrict__ o2GlobalPtr,
                  const T *__restrict__ o12GTGlobalPtr,
                  T *__restrict__ lossGlobalPtr, T *__restrict__ do1GlobalPtr,
                  T *__restrict__ do2GlobalPtr, int numImagePairs) {

  const int threadIdxInBlock = threadIdx.x;
  const int numThreadsInBlock = blockDim.x;
  const int blockStride = gridDim.x * BATCH_SIZE;

  // Allocate shared memory
  __shared__ T sharedBuffer3x1a[BATCH_SIZE][3];
  __shared__ T sharedBuffer3x1b[BATCH_SIZE][3];
  __shared__ T sharedBuffer3x1c[BATCH_SIZE][3];
  __shared__ T sharedBuffer1x1a[BATCH_SIZE]; // shared memory for scalar values
  __shared__ T sharedBuffer1x1b[BATCH_SIZE]; // shared memory for scalar values

  // Loop until all batches for the block are processed
  for (int startImagePairIdx = blockIdx.x * BATCH_SIZE;
       startImagePairIdx < numImagePairs; startImagePairIdx += blockStride) {
    // Determine the actual batch size to account for the last batch
    int numImagePairsInBatch = BATCH_SIZE;
    if (startImagePairIdx + BATCH_SIZE > numImagePairs) {
      numImagePairsInBatch = numImagePairs - startImagePairIdx;
    }
    int pairIdxInBatch = threadIdxInBlock / 3;
    int colIdxInBatch = threadIdxInBlock % 3;

    // Assume we have at least BATCH_SIZE * 3 threads in the block
    int minNumThreadsInBlock = BATCH_SIZE * 3;
    assert(
        numThreadsInBlock >= minNumThreadsInBlock &&
        "Number of threads in the block must be at least BATCH_SIZE * 3 * 3");

    // Load o1 and o2 into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(o1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         reinterpret_cast<char *>(O1),
                         numImagePairsInBatch * sizeof(T) * 3);
    copyContiguousMemory(reinterpret_cast<const char *>(o2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         reinterpret_cast<char *>(O2),
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();

    // Compute o12 = o2 - o1
    if (threadIdxInBlock < minNumThreadsInBlock) {
      O12[pairIdxInBatch][colIdxInBatch] =
          O2[pairIdxInBatch][colIdxInBatch] - O1[pairIdxInBatch][colIdxInBatch];
    }
    __syncthreads();

    // Compute norm of o12
    if (threadIdxInBlock < minNumThreadsInBlock && colIdxInBatch == 0) {
      O12_NORM[pairIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        O12_NORM[pairIdxInBatch] +=
            O12[pairIdxInBatch][i] * O12[pairIdxInBatch][i];
      }
      O12_NORM[pairIdxInBatch] =
          sqrt(O12_NORM[pairIdxInBatch]) + 1e-12; // Avoid division by zero
    }
    __syncthreads();

    // Normalize o12
    if (threadIdxInBlock < minNumThreadsInBlock) {
      O12_NORMALIZED[pairIdxInBatch][colIdxInBatch] =
          O12[pairIdxInBatch][colIdxInBatch] / O12_NORM[pairIdxInBatch];
    }
    __syncthreads();

    // Load o12GT into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(o12GTGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         reinterpret_cast<char *>(O12_GT),
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();

    // Compute and write loss
    if (pairIdxInBatch < minNumThreadsInBlock && colIdxInBatch == 0) {
      LOSS[pairIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        LOSS[pairIdxInBatch] +=
            abs(O12_NORMALIZED[pairIdxInBatch][i] - O12_GT[pairIdxInBatch][i]);
      }
    }
    __syncthreads();
    if (pairIdxInBatch == 0 && colIdxInBatch == 0) {
      T totalLoss = 0;
      for (int i = 0; i < numImagePairsInBatch; i++) {
        totalLoss += LOSS[i];
      }
      atomicAdd(lossGlobalPtr,
                totalLoss / (static_cast<T>(numImagePairs) * 3.0));
    }
    __syncthreads();

    // Compute d_o12_normalized
    if (threadIdxInBlock < minNumThreadsInBlock) {
      d_O12_NORMALIZED[pairIdxInBatch][colIdxInBatch] =
          O12_NORMALIZED[pairIdxInBatch][colIdxInBatch] >
                  O12_GT[pairIdxInBatch][colIdxInBatch]
              ? 1.0
              : -1.0;
      d_O12_NORMALIZED[pairIdxInBatch][colIdxInBatch] /=
          static_cast<T>(numImagePairs) * 3.0;
    }
    __syncthreads();

    // Compute d_o12
    if (threadIdxInBlock < minNumThreadsInBlock) {
      T dot = 0;
      for (int i = 0; i < 3; i++) {
        dot += O12_NORMALIZED[pairIdxInBatch][i] *
               d_O12_NORMALIZED[pairIdxInBatch][i];
      }
      d_O12[pairIdxInBatch][colIdxInBatch] =
          (d_O12_NORMALIZED[pairIdxInBatch][colIdxInBatch] -
           dot * O12_NORMALIZED[pairIdxInBatch][colIdxInBatch]) /
          O12_NORM[pairIdxInBatch];
    }
    __syncthreads();

    // Write d_o2
    copyContiguousMemory(reinterpret_cast<const char *>(d_O2),
                         reinterpret_cast<char *>(do2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();

    // Compute and write d_o1
    if (threadIdxInBlock < minNumThreadsInBlock) {
      d_O1[pairIdxInBatch][colIdxInBatch] =
          -d_O12[pairIdxInBatch][colIdxInBatch];
    }
    __syncthreads();
    copyContiguousMemory(reinterpret_cast<const char *>(d_O1),
                         reinterpret_cast<char *>(do1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();
  }
}

// Thin C++ wrapper that launches the kernel on the current stream
template <typename T>
void translation_gradient(const at::Tensor &o1, const at::Tensor &o2,
                          const at::Tensor &o12GT, at::Tensor &loss,
                          at::Tensor &do1, at::Tensor &do2) {
  TORCH_CHECK(o1.is_cuda() && o2.is_cuda() && o12GT.is_cuda() &&
                  loss.is_cuda() && do1.is_cuda() && do2.is_cuda(),
              "All tensors must be on CUDA");
  int numImagePairs =
      o1.size(0) *
      o1.size(1); // Note that o1 has size (num_init, num_image_pairs, 3)

  constexpr int numThreadsPerBlock = BATCH_SIZE * 3;
  const int numBlocks = 1024;

  // Set loss tensor to zero
  loss.zero_();

  // Launch the kernel
  translationKernel<T>
      <<<numBlocks, numThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          o1.data_ptr<T>(), o2.data_ptr<T>(), o12GT.data_ptr<T>(),
          loss.data_ptr<T>(), do1.data_ptr<T>(), do2.data_ptr<T>(),
          numImagePairs);

  // Synchronize to ensure kernel execution is complete
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in translation_gradient kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
  cudaDeviceSynchronize();
}

// Explicit instantiation
template void translation_gradient<float>(const at::Tensor &o1,
                                          const at::Tensor &o2,
                                          const at::Tensor &o12GT,
                                          at::Tensor &loss, at::Tensor &do1,
                                          at::Tensor &do2);

#include "kernels.h"
#include <cassert>
#include <cmath>

// define aliases for shared memory buffers
#define R_REL sharedBuffer3x3a
#define R_W2C_1 sharedBuffer3x3b
#define R1 sharedBuffer3x3c
#define R2                                                                     \
  sharedBuffer3x3b // Make sure R_W2C_1 is can be overwritten at this point
#define TRACE sharedBuffer1x1a
#define COS sharedBuffer1x1a         // Will overwrite TRACE
#define COS_CLAMPED sharedBuffer1x1a // Will overwrite COS
#define CLAMP_MASK sharedBuffer1x1b
#define d_TRACE sharedBuffer1x1a   // Will overwrite COS_CLAMPED
#define d_R_W2C_1 sharedBuffer3x3d // Will overwrite R1
#define d_R_W2C_2 sharedBuffer3x3d // Will overwrite d_R_W2C_1

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
rotationKernel(const T *__restrict__ RrelGlobalPtr,
               const T *__restrict__ Rw2c1GlobalPtr,
               const T *__restrict__ Rw2c2GlobalPtr,
               T *__restrict__ lossGlobalPtr, T *__restrict__ dRw2c1GlobalPtr,
               T *__restrict__ dRw2c2GlobalPtr, T clampThr, int numImagePairs) {

  const int threadIdxInBlock = threadIdx.x;
  const int numThreadsInBlock = blockDim.x;
  const int blockStride = gridDim.x * BATCH_SIZE;

  // Allocate shared memory
  __shared__ T
      sharedBuffer3x3a[BATCH_SIZE][3][3]; // shared memory for 3x3 matrices
  __shared__ T
      sharedBuffer3x3b[BATCH_SIZE][3][3]; // shared memory for 3x3 matrices
  __shared__ T
      sharedBuffer3x3c[BATCH_SIZE][3][3]; // shared memory for 3x3 matrices
  __shared__ T
      sharedBuffer3x3d[BATCH_SIZE][3][3];    // shared memory for 3x3 matrices
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
    int pairIdxInBatch = threadIdxInBlock / 9;
    int rowIdxInBatch = (threadIdxInBlock % 9) / 3;
    int colIdxInBatch = (threadIdxInBlock % 9) % 3;

    // Assume we have at least BATCH_SIZE * 3 * 3 threads in the block
    int minNumThreadsInBlock = BATCH_SIZE * 3 * 3;
    assert(
        numThreadsInBlock >= minNumThreadsInBlock &&
        "Number of threads in the block must be at least BATCH_SIZE * 3 * 3");

    // Load R_rel into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(RrelGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(R_REL),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();

    // Compute R1 = R_rel @ Rw2c1
    copyContiguousMemory(reinterpret_cast<const char *>(Rw2c1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(R_W2C_1),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    if (threadIdxInBlock < minNumThreadsInBlock) {
      R1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        R1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            R_REL[pairIdxInBatch][rowIdxInBatch][i] *
            R_W2C_1[pairIdxInBatch][i][colIdxInBatch];
      }
    }
    __syncthreads();

    // Copy R2 = Rw2c2
    copyContiguousMemory(reinterpret_cast<const char *>(Rw2c2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(R2),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();

    // Compute clamped cosine
    if (threadIdxInBlock < minNumThreadsInBlock && rowIdxInBatch == 0 &&
        colIdxInBatch == 0) {
      TRACE[pairIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          TRACE[pairIdxInBatch] +=
              R1[pairIdxInBatch][i][j] * R2[pairIdxInBatch][i][j];
        }
      }
      COS[pairIdxInBatch] = (TRACE[pairIdxInBatch] - 1) * 0.5;
      CLAMP_MASK[pairIdxInBatch] =
          (COS[pairIdxInBatch] > -clampThr && COS[pairIdxInBatch] < clampThr)
              ? 1
              : 0;
      COS_CLAMPED[pairIdxInBatch] =
          CLAMP_MASK[pairIdxInBatch] * COS[pairIdxInBatch] +
          (1 - CLAMP_MASK[pairIdxInBatch]) * clampThr *
              ((COS[pairIdxInBatch] > 0) ? 1 : -1);
    }
    __syncthreads();

    // Compute and write loss
    if (threadIdxInBlock < minNumThreadsInBlock &&
        pairIdxInBatch < numImagePairsInBatch && rowIdxInBatch == 0 &&
        colIdxInBatch == 0) {
      atomicAdd(lossGlobalPtr, acos(COS_CLAMPED[pairIdxInBatch]) /
                                   static_cast<T>(numImagePairs));
    }
    __syncthreads();

    // Compute d_TRACE
    if (threadIdxInBlock < minNumThreadsInBlock && rowIdxInBatch == 0 &&
        colIdxInBatch == 0) {
      d_TRACE[pairIdxInBatch] = -(1.0 / static_cast<T>(numImagePairs)) /
                                sqrt(1.0 - COS_CLAMPED[pairIdxInBatch] *
                                               COS_CLAMPED[pairIdxInBatch]) *
                                CLAMP_MASK[pairIdxInBatch] * 0.5;
    }
    __syncthreads();

    // Compute and write d_R_w2c1
    if (threadIdxInBlock < minNumThreadsInBlock) {
      d_R_W2C_1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        d_R_W2C_1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            d_TRACE[pairIdxInBatch] * R_REL[pairIdxInBatch][i][rowIdxInBatch] *
            R2[pairIdxInBatch][i][colIdxInBatch];
      }
    }
    __syncthreads();
    copyContiguousMemory(reinterpret_cast<const char *>(d_R_W2C_1),
                         reinterpret_cast<char *>(dRw2c1GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();

    // Compute and write d_R_w2c2
    if (threadIdxInBlock < minNumThreadsInBlock) {
      d_R_W2C_2[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          d_TRACE[pairIdxInBatch] *
          R1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch];
    }
    __syncthreads();
    copyContiguousMemory(reinterpret_cast<const char *>(d_R_W2C_2),
                         reinterpret_cast<char *>(dRw2c2GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
  }
}

// Thin C++ wrapper that launches the kernel on the current stream
template <typename T>
void rotation_gradient(const at::Tensor &Rrel, const at::Tensor &Rw2c1,
                       const at::Tensor &Rw2c2, at::Tensor &loss,
                       at::Tensor &dRw2c1, at::Tensor &dRw2c2, T clampThr) {
  TORCH_CHECK(Rrel.is_cuda() && Rw2c1.is_cuda() && Rw2c2.is_cuda(),
              "Tensors must be on CUDA");
  int numImagePairs = Rrel.size(0);

  constexpr int numThreadsPerBlock = BATCH_SIZE * 3 * 3;
  const int numBlocks = 1024;

  // Set loss tensor to zero
  loss.zero_();

  // Launch the kernel
  rotationKernel<T>
      <<<numBlocks, numThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          Rrel.data_ptr<T>(), Rw2c1.data_ptr<T>(), Rw2c2.data_ptr<T>(),
          loss.data_ptr<T>(), dRw2c1.data_ptr<T>(), dRw2c2.data_ptr<T>(),
          clampThr, numImagePairs);

  // Synchronize to ensure kernel execution is complete
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in rotation_gradient kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
  cudaDeviceSynchronize();
}

// Explicit instantiation
template void rotation_gradient<float>(const at::Tensor &Rrel,
                                       const at::Tensor &Rw2c1,
                                       const at::Tensor &Rw2c2,
                                       at::Tensor &loss, at::Tensor &dRw2c1,
                                       at::Tensor &dRw2c2, float clampThr);

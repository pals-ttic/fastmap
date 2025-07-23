#include "kernels.h"
#include <cassert>

// define aliases for shared memory buffers
#define OP1_R1 sharedBuffer3x3a
#define OP1_R2 sharedBuffer3x3b
#define OP1_R_REL sharedBuffer3x3c

constexpr int WARP_SIZE = 32;
constexpr int BATCH_SIZE = 8; // Adjust as needed

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
__global__ void epipolarKernel(const T *__restrict__ R1GlobalPtr,
                               const T *__restrict__ R2GlobalPtr,
                               const T *__restrict__ t1GlobalPtr,
                               const T *__restrict__ t2GlobalPtr,
                               const T *__restrict__ WGlobalPtr,
                               T *__restrict__ outputGlobalPtr,
                               int numImagePairs) {

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

  // Loop until all batches for the block are processed
  for (int startImagePairIdx = blockIdx.x * BATCH_SIZE;
       startImagePairIdx < numImagePairs; startImagePairIdx += blockStride) {
    // Determine the actual batch size to account for the last batch
    int numImagePairsInBatch = BATCH_SIZE;
    if (startImagePairIdx + BATCH_SIZE > numImagePairs) {
      numImagePairsInBatch = numImagePairs - startImagePairIdx;
    }

    // -------- Op 1: Compute R_rel --------
    // alias buffers for shared memory
    // Load R1 and R2 into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(R1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(OP1_R1),
                         numImagePairsInBatch * sizeof(T) * 9);
    copyContiguousMemory(reinterpret_cast<const char *>(R2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(OP1_R2),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    // Compute R_rel = R2 * R1^T
    // Assume we have at least BATCH_SIZE * 3 * 3 threads in the block
    assert(numThreadsInBlock >= BATCH_SIZE * 3 * 3);
    if (threadIdxInBlock < numImagePairsInBatch * 9) {
      int pairIdxInBatch = threadIdxInBlock / 9;
      int rowIdxInBatch = (threadIdxInBlock % 9) / 3;
      int colIdxInBatch = (threadIdxInBlock % 9) % 3;
      OP1_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        OP1_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            OP1_R2[pairIdxInBatch][rowIdxInBatch][i] *
            OP1_R1[pairIdxInBatch][colIdxInBatch]
                  [i]; // note that R1 is transposed
      }
    }
    __syncthreads();
    // Copy R_rel to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(OP1_R_REL),
                         reinterpret_cast<char *>(outputGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
  }
}

// Thin C++ wrapper that launches the kernel on the current stream
template <typename T>
at::Tensor epipolar_gradient(const at::Tensor &R1, const at::Tensor &R2,
                             const at::Tensor &t1, const at::Tensor &t2,
                             const at::Tensor &W) {
  TORCH_CHECK(R1.is_cuda() && R2.is_cuda() && t1.is_cuda() && t2.is_cuda() &&
                  W.is_cuda(),
              "Tensors must be on CUDA");
  auto R_rel = at::empty_like(R1);

  constexpr int numThreadsPerBlock = BATCH_SIZE * 3 * 3;
  const int numBlocks = 1024;

  // Launch the kernel
  epipolarKernel<T>
      <<<numBlocks, numThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          R1.data_ptr<T>(), R2.data_ptr<T>(), t1.data_ptr<T>(),
          t2.data_ptr<T>(), W.data_ptr<T>(), R_rel.data_ptr<T>(), R1.size(0));

  return R_rel;
}

// Explicit instantiation
template at::Tensor epipolar_gradient<float>(const at::Tensor &R1,
                                             const at::Tensor &R2,
                                             const at::Tensor &t1,
                                             const at::Tensor &t2,
                                             const at::Tensor &W);

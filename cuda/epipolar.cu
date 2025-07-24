#include "kernels.h"
#include <cassert>

// define aliases for shared memory buffers
// --- Layer 1 ---
#define LAYER1_R1 sharedBuffer3x3a
#define LAYER1_R2 sharedBuffer3x3b
#define LAYER1_R_REL sharedBuffer3x3c
// --- Layer 2 ---
#define LAYER2_R_REL sharedBuffer3x3c // re-use the previous buffer for R_rel
#define LAYER2_T1X sharedBuffer3x3a
#define LAYER2_T2X sharedBuffer3x3a
#define LAYER2_ESSENTIAL sharedBuffer3x3b
#define LAYER2_T1 sharedBuffer3x1a
#define LAYER2_T2 sharedBuffer3x1b

constexpr int WARP_SIZE = 32;
constexpr int BATCH_SIZE = 8; // Adjust as needed
__constant__ int SKEW_SYMMETRIC_IDX[3][3]{
    {0, 2, 1},
    {2, 1, 0},
    {1, 0, 2}}; // indices for skew-symmetric matrix (diagonal is arbitrary)
__constant__ float SKEW_SYMMETRIC_SIGN[3][3]{
    {0, -1, 1},
    {1, 0, -1},
    {-1, 1, 0}}; // Sign matrix for skew-symmetric matrix

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
__global__ void epipolarKernel(
    const T *__restrict__ R1GlobalPtr, const T *__restrict__ R2GlobalPtr,
    const T *__restrict__ t1GlobalPtr, const T *__restrict__ t2GlobalPtr,
    const T *__restrict__ WGlobalPtr, T *__restrict__ RrelGlobalPtr,
    T *__restrict__ t1xGlobalPtr, T *__restrict__ t2xGlobalPtr,
    T *__restrict__ outputGlobalPtr, int numImagePairs) {

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
  __shared__ T sharedBuffer3x1a[BATCH_SIZE]
                               [3]; // shared memory for 3-dimensional vectors
  __shared__ T sharedBuffer3x1b[BATCH_SIZE]
                               [3]; // shared memory for 3-dimensional vectors

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

    // -------- Layer 1: Compute R_rel --------
    // Load R1 and R2 into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(R1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(LAYER1_R1),
                         numImagePairsInBatch * sizeof(T) * 9);
    copyContiguousMemory(reinterpret_cast<const char *>(R2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 9,
                         reinterpret_cast<char *>(LAYER1_R2),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    // Compute R_rel = R2 * R1^T
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER1_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        LAYER1_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER1_R2[pairIdxInBatch][rowIdxInBatch][i] *
            LAYER1_R1[pairIdxInBatch][colIdxInBatch]
                     [i]; // note that R1 is transposed
      }
    }
    __syncthreads();
    // Copy R_rel to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER1_R_REL),
                         reinterpret_cast<char *>(RrelGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);

    // -------- Layer 2: Compute essential --------
    // Load t1 into shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(t1GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         reinterpret_cast<char *>(LAYER2_T1),
                         numImagePairsInBatch * sizeof(T) * 3);
    copyContiguousMemory(reinterpret_cast<const char *>(t2GlobalPtr) +
                             startImagePairIdx * sizeof(T) * 3,
                         reinterpret_cast<char *>(LAYER2_T2),
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();
    // Clear memory for essential matrix
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER2_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
    }
    // Form skew-symmetric matrices for t1
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER2_T1X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          SKEW_SYMMETRIC_SIGN[rowIdxInBatch][colIdxInBatch] *
          LAYER2_T1[pairIdxInBatch]
                   [SKEW_SYMMETRIC_IDX[rowIdxInBatch][colIdxInBatch]];
    }
    __syncthreads();
    // Write t1_x to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER2_T1X),
                         reinterpret_cast<char *>(t1xGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    // Accumulate to the essential matrix with t1_x
    if (threadIdxInBlock < minNumThreadsInBlock) {
      for (int i = 0; i < 3; i++) {
        LAYER2_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER2_R_REL[pairIdxInBatch][rowIdxInBatch][i] *
            LAYER2_T1X[pairIdxInBatch][i][colIdxInBatch];
      }
    }
    __syncthreads();
    // Form skew-symmetric matrices for t2
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER2_T2X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          SKEW_SYMMETRIC_SIGN[rowIdxInBatch][colIdxInBatch] *
          LAYER2_T2[pairIdxInBatch]
                   [SKEW_SYMMETRIC_IDX[rowIdxInBatch][colIdxInBatch]];
    }
    __syncthreads();
    // Write t2_x to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER2_T2X),
                         reinterpret_cast<char *>(t2xGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    // Accumulate to the essential matrix with t2_x
    if (threadIdxInBlock < minNumThreadsInBlock) {
      for (int i = 0; i < 3; i++) {
        LAYER2_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] -=
            LAYER2_T2X[pairIdxInBatch][rowIdxInBatch][i] *
            LAYER2_R_REL[pairIdxInBatch][i][colIdxInBatch];
      }
    }
    __syncthreads();
    // Copy essential to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER2_ESSENTIAL),
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
  auto out = at::empty_like(R1);
  int numImagePairs = R1.size(0);

  constexpr int numThreadsPerBlock = BATCH_SIZE * 3 * 3;
  const int numBlocks = 1024;

  // Allocae buffers on HBM
  T *RrelGlobalPtr; // Global pointer for R_rel
  T *t1xGlobalPtr;  // Global pointer for [t1]_x
  T *t2xGlobalPtr;  // Global pointer for [t2]_x
  cudaMalloc(&RrelGlobalPtr, numImagePairs * sizeof(T) * 9);
  cudaMalloc(&t1xGlobalPtr, numImagePairs * sizeof(T) * 9);
  cudaMalloc(&t2xGlobalPtr, numImagePairs * sizeof(T) * 9);

  // Launch the kernel
  epipolarKernel<T>
      <<<numBlocks, numThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          R1.data_ptr<T>(), R2.data_ptr<T>(), t1.data_ptr<T>(),
          t2.data_ptr<T>(), W.data_ptr<T>(), RrelGlobalPtr, t1xGlobalPtr,
          t2xGlobalPtr, out.data_ptr<T>(), numImagePairs);

  return out;
}

// Explicit instantiation
template at::Tensor epipolar_gradient<float>(const at::Tensor &R1,
                                             const at::Tensor &R2,
                                             const at::Tensor &t1,
                                             const at::Tensor &t2,
                                             const at::Tensor &W);

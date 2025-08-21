#include "kernels.h"
#include <cassert>
#include <tuple>

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
// --- Layer 3 ---
#define LAYER3_F1_INV sharedBuffer1x1a
#define LAYER3_F2_INV sharedBuffer1x1b
#define LAYER3_ESSENTIAL                                                       \
  sharedBuffer3x3b // re-use the previous buffer for essential matrix
#define LAYER3_FUNDAMENTAL                                                     \
  sharedBuffer3x3b // inplace overwrite essential matrix
#define LAYER3_FUNDAMENTAL_NORM                                                \
  sharedBuffer1x1a // buffer for fundamental norm (overwrite f1_inv)
// --- Layer 4 ---
#define LAYER4_FUNDAMENTAL sharedBuffer3x3b
#define LAYER4_d_FUNDAMENTAL sharedBuffer3x3a
#define LAYER4_TEMP                                                            \
  sharedBuffer3x3c // temporary buffer to store the element wise product of
                   // fundamental and W @ fundamental
#define LAYER4_LOSS                                                            \
  sharedBuffer3x3c[0][0][0] // the first element of this buffer will
                            // contain the loss value
// --- Layer 5 ---
#define LAYER5_FUNDAMENTAL sharedBuffer3x3b
#define LAYER5_d_FUNDAMENTAL sharedBuffer3x3a
#define LAYER5_FUNDAMENTAL_NORM                                                \
  sharedBuffer1x1a // re-used from Layer 3 for fundamental norm
#define LAYER5_TEMP                                                            \
  sharedBuffer1x1b // temporary buffer to store the dot product of F and d_F
#define LAYER5_d_UNNORMALIZED_FUNDAMENTAL sharedBuffer3x3c
// --- Layer 6 ---
#define LAYER6_d_UNNORMALIZED_FUNDAMENTAL sharedBuffer3x3c
#define LAYER6_ESSENTIAL sharedBuffer3x3b
#define LAYER6_d_K1_INV sharedBuffer3x3a
#define LAYER6_F1_INV sharedBuffer1x1a
#define LAYER6_d_F1_INV sharedBuffer1x1b
#define LAYER6_d_K2_INV sharedBuffer3x3a // re-use buffer for f1 and f2
#define LAYER6_F2_INV sharedBuffer1x1a   // re use buffer for f1 and f2
#define LAYER6_d_F2_INV sharedBuffer1x1b // re use buffer for f1 and f2
// --- Layer 7 ---
#define LAYER7_d_UNNORMALIZED_FUNDAMENTAL sharedBuffer3x3c
#define LAYER7_d_ESSENTIAL sharedBuffer3x3b
#define LAYER7_F1_INV sharedBuffer1x1a
#define LAYER7_F2_INV sharedBuffer1x1b
// --- Layer 8 ---
#define LAYER8_d_ESSENTIAL sharedBuffer3x3b
#define LAYER8_R_REL sharedBuffer3x3c
#define LAYER8_d_T1X sharedBuffer3x3a
#define LAYER8_d_T2X sharedBuffer3x3a // re-use buffer for t1_x and t2_x
#define LAYER8_d_T1 sharedBuffer3x1a
#define LAYER8_d_T2 sharedBuffer3x1a // re-use buffer for d_t1 and d_t2
// --- Layer 9 ---
#define LAYER9_d_ESSENTIAL sharedBuffer3x3b
#define LAYER9_d_R_REL sharedBuffer3x3a
#define LAYER9_T1X sharedBuffer3x3c
#define LAYER9_T2X sharedBuffer3x3c // re-use buffer for t1_x and t2_x
#define LAYER9_R2 sharedBuffer3x3b
#define LAYER9_d_R1 sharedBuffer3x3c
#define LAYER9_R1 sharedBuffer3x3b   // re-use buffer for R1 and R2
#define LAYER9_d_R2 sharedBuffer3x3c // re-use buffer for d_R1 and d_R2

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

// Reduction function to sum elements in a shared memory array and store the
// result at the first element.
template <typename T>
__device__ __forceinline__ void reduceSum(T *__restrict__ startPtr,
                                          int numElements) {
  int stride = (numElements + 1) / 2;
  while (numElements > 1) {
    if (threadIdx.x < stride && threadIdx.x + stride < numElements) {
      startPtr[threadIdx.x] += startPtr[threadIdx.x + stride];
    }
    __syncthreads();
    numElements = stride;
    stride = (numElements + 1) / 2;
  }
  return;
}

template <typename T>
__global__ void epipolarKernel(
    const T *__restrict__ R1GlobalPtr, const T *__restrict__ R2GlobalPtr,
    const T *__restrict__ t1GlobalPtr, const T *__restrict__ t2GlobalPtr,
    const T *__restrict__ f1InvGlobalPtr, const T *__restrict__ f2InvGlobalPtr,
    const T *__restrict__ WGlobalPtr, T *__restrict__ RrelGlobalPtr,
    T *__restrict__ t1xGlobalPtr, T *__restrict__ t2xGlobalPtr,
    T *__restrict__ essentialGlobalPtr, T *__restrict__ FundamentalGlobalPtr,
    T *__restrict__ lossGlobalPtr, T *__restrict__ dR1GlobalPtr,
    T *__restrict__ dR2GlobalPtr, T *__restrict__ dt1GlobalPtr,
    T *__restrict__ dt2GlobalPtr, T *__restrict__ dF1InvGlobalPtr,
    T *__restrict__ dF2InvGlobalPtr, int numImagePairs) {

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
  __shared__ T
      sharedBuffer1x1a[BATCH_SIZE]; // shared memory for 3-dimensional vectors
  __shared__ T
      sharedBuffer1x1b[BATCH_SIZE]; // shared memory for 3-dimensional vectors

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
                         reinterpret_cast<char *>(essentialGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();

    // -------- Layer 3: Compute fundamental --------
    // Load focal lengths from global memory
    copyContiguousMemory(reinterpret_cast<const char *>(f1InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER3_F1_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    copyContiguousMemory(reinterpret_cast<const char *>(f2InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER3_F2_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    __syncthreads();
    // Compute unnormalized fundamental matrix
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (colIdxInBatch < 2) {
        LAYER3_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] *=
            LAYER3_F1_INV[pairIdxInBatch];
      }
      if (rowIdxInBatch < 2) {
        LAYER3_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] *=
            LAYER3_F2_INV[pairIdxInBatch];
      }
    }
    __syncthreads();
    // Compute fundamental matrix norm
    if (threadIdxInBlock < numThreadsInBlock) {
      if (rowIdxInBatch == 0 && colIdxInBatch == 0) {
        LAYER3_FUNDAMENTAL_NORM[pairIdxInBatch] = 0;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            LAYER3_FUNDAMENTAL_NORM[pairIdxInBatch] +=
                LAYER3_FUNDAMENTAL[pairIdxInBatch][i][j] *
                LAYER3_FUNDAMENTAL[pairIdxInBatch][i][j];
          }
        }
        LAYER3_FUNDAMENTAL_NORM[pairIdxInBatch] =
            sqrt(LAYER3_FUNDAMENTAL_NORM[pairIdxInBatch]) + 1e-8;
      }
    }
    __syncthreads();
    // Normalize the fundamental matrix
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER3_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] /=
          LAYER3_FUNDAMENTAL_NORM[pairIdxInBatch];
    }
    __syncthreads();
    // Copy fundamental matrix to global memory
    // copyContiguousMemory(reinterpret_cast<const char *>(LAYER3_FUNDAMENTAL),
    //                      reinterpret_cast<char *>(FundamentalGlobalPtr) +
    //                          startImagePairIdx * 9 * sizeof(T),
    //                      numImagePairsInBatch * sizeof(T) * 9);
    // __syncthreads();

    // -------- Layer 4: Compute loss --------
    // Compute W @ fundamental (which is d_fundamental)
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER4_d_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          LAYER4_d_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
              WGlobalPtr[(startImagePairIdx + pairIdxInBatch) * 81 +
                         (rowIdxInBatch * 3 + colIdxInBatch) * 9 +
                         (i * 3 + j)] *
              LAYER4_FUNDAMENTAL[pairIdxInBatch][i][j];
        }
      }
    }
    __syncthreads();
    // Compute element-wise product of fundamental and W @ fundamental
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER4_TEMP[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          0.5 *
          LAYER4_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] *
          LAYER4_d_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch];
    }
    __syncthreads();
    // Reduction to compute the loss
    reduceSum<T>(&LAYER4_TEMP[0][0][0], numImagePairsInBatch * 9);
    __syncthreads();
    // write the loss the global memory
    if (threadIdxInBlock == 0) {
      atomicAdd(lossGlobalPtr,
                LAYER4_TEMP[0][0][0]); // atomic add to the loss
    }
    __syncthreads();

    // -------- Layer 5: Compute d_unnormalized_fundamental --------
    // Compute dot product of fundamental and d_fundamental
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (rowIdxInBatch == 0 && colIdxInBatch == 0) {
        LAYER5_TEMP[pairIdxInBatch] = 0;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            LAYER5_TEMP[pairIdxInBatch] +=
                LAYER5_FUNDAMENTAL[pairIdxInBatch][i][j] *
                LAYER5_d_FUNDAMENTAL[pairIdxInBatch][i][j];
          }
        }
      }
    }
    __syncthreads();
    // Compute gradient with respect to unnormalized fundamental matrix
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER5_d_UNNORMALIZED_FUNDAMENTAL
          [pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
              (LAYER5_d_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch]
                                   [colIdxInBatch] -
               LAYER5_TEMP[pairIdxInBatch] *
                   LAYER5_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch]
                                     [colIdxInBatch]) /
              LAYER5_FUNDAMENTAL_NORM[pairIdxInBatch];
    }
    __syncthreads();

    // -------- Layer 6: Compute d_f1_inv, d_f2_inv --------
    // Copy essential to shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(essentialGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER6_ESSENTIAL),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    // copy f2_inv to shared memory (for computing d_f1_inv)
    copyContiguousMemory(reinterpret_cast<const char *>(f2InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER6_F2_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    __syncthreads();
    // Compute d_K1_inv
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER6_d_K1_INV[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          LAYER6_d_UNNORMALIZED_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch]
                                           [colIdxInBatch] *
          LAYER6_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] *
          (rowIdxInBatch == 2 ? 1.0 : LAYER6_F2_INV[pairIdxInBatch]);
    }
    __syncthreads();
    // Compute d_f1_inv
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (rowIdxInBatch == 0 && colIdxInBatch == 0) {
        LAYER6_d_F1_INV[pairIdxInBatch] = 0;
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 2; j++) {
            LAYER6_d_F1_INV[pairIdxInBatch] +=
                LAYER6_d_K1_INV[pairIdxInBatch][i][j];
          }
        }
      }
    }
    __syncthreads();
    // Write d_f1_inv to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER6_d_F1_INV),
                         reinterpret_cast<char *>(dF1InvGlobalPtr) +
                             startImagePairIdx * sizeof(T),
                         numImagePairsInBatch * sizeof(T));
    __syncthreads();
    // copy f1_inv to shared memory (for computing d_f2_inv)
    copyContiguousMemory(reinterpret_cast<const char *>(f1InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER6_F1_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    __syncthreads();
    // Compute d_K2_inv
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER6_d_K2_INV[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          LAYER6_d_UNNORMALIZED_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch]
                                           [colIdxInBatch] *
          LAYER6_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] *
          (colIdxInBatch == 2 ? 1.0 : LAYER6_F1_INV[pairIdxInBatch]);
    }
    __syncthreads();
    // Compute d_f2_inv
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (rowIdxInBatch == 0 && colIdxInBatch == 0) {
        LAYER6_d_F2_INV[pairIdxInBatch] = 0;
        for (int i = 0; i < 2; i++) {
          for (int j = 0; j < 3; j++) {
            LAYER6_d_F2_INV[pairIdxInBatch] +=
                LAYER6_d_K2_INV[pairIdxInBatch][i][j];
          }
        }
      }
    }
    __syncthreads();
    // Write d_f2_inv to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER6_d_F2_INV),
                         reinterpret_cast<char *>(dF2InvGlobalPtr) +
                             startImagePairIdx * sizeof(T),
                         numImagePairsInBatch * sizeof(T));
    __syncthreads();

    // -------- Layer 7: Compute d_essential --------
    // copy f1_inv and f2_inv to shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(f1InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER7_F1_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    copyContiguousMemory(reinterpret_cast<const char *>(f2InvGlobalPtr) +
                             startImagePairIdx * sizeof(T) * 1,
                         reinterpret_cast<char *>(LAYER7_F2_INV),
                         numImagePairsInBatch * sizeof(T) * 1);
    __syncthreads();
    // compute d_essential
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER7_d_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] =
          LAYER7_d_UNNORMALIZED_FUNDAMENTAL[pairIdxInBatch][rowIdxInBatch]
                                           [colIdxInBatch] *
          (colIdxInBatch == 2 ? 1.0 : LAYER7_F1_INV[pairIdxInBatch]) *
          (rowIdxInBatch == 2 ? 1.0 : LAYER7_F2_INV[pairIdxInBatch]);
    }
    __syncthreads();

    // -------- Layer 8: Compute d_t1, d_t2 --------
    // Copy R_rel to shared memory
    copyContiguousMemory(reinterpret_cast<const char *>(RrelGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER8_R_REL),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    // Compute d_t1_x
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER8_d_T1X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        LAYER8_d_T1X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER8_R_REL[pairIdxInBatch][i][rowIdxInBatch] *
            LAYER8_d_ESSENTIAL[pairIdxInBatch][i]
                              [colIdxInBatch]; // Note that R_rel is transposed
      }
    }
    __syncthreads();
    // Compute d_t1
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (rowIdxInBatch == 0) {
        if (colIdxInBatch == 0) {
          LAYER8_d_T1[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T1X[pairIdxInBatch][2][1] -
              LAYER8_d_T1X[pairIdxInBatch][1][2];
        }
        if (colIdxInBatch == 1) {
          LAYER8_d_T1[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T1X[pairIdxInBatch][0][2] -
              LAYER8_d_T1X[pairIdxInBatch][2][0];
        }
        if (colIdxInBatch == 2) {
          LAYER8_d_T1[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T1X[pairIdxInBatch][1][0] -
              LAYER8_d_T1X[pairIdxInBatch][0][1];
        }
      }
    }
    __syncthreads();
    // Write d_t1 to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER8_d_T1),
                         reinterpret_cast<char *>(dt1GlobalPtr) +
                             startImagePairIdx * 3 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();
    // Compute d_t1_x
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER8_d_T2X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int j = 0; j < 3; j++) {
        LAYER8_d_T2X[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] -=
            LAYER8_d_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][j] *
            LAYER8_R_REL[pairIdxInBatch][colIdxInBatch]
                        [j]; // Note that R_rel is transposed
      }
    }
    __syncthreads();
    // Compute d_t2
    if (threadIdxInBlock < minNumThreadsInBlock) {
      if (rowIdxInBatch == 0) {
        if (colIdxInBatch == 0) {
          LAYER8_d_T2[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T2X[pairIdxInBatch][2][1] -
              LAYER8_d_T2X[pairIdxInBatch][1][2];
        }
        if (colIdxInBatch == 1) {
          LAYER8_d_T2[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T2X[pairIdxInBatch][0][2] -
              LAYER8_d_T2X[pairIdxInBatch][2][0];
        }
        if (colIdxInBatch == 2) {
          LAYER8_d_T2[pairIdxInBatch][colIdxInBatch] =
              LAYER8_d_T2X[pairIdxInBatch][1][0] -
              LAYER8_d_T2X[pairIdxInBatch][0][1];
        }
      }
    }
    __syncthreads();
    // Write d_t2 to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER8_d_T2),
                         reinterpret_cast<char *>(dt2GlobalPtr) +
                             startImagePairIdx * 3 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 3);
    __syncthreads();

    // -------- Layer 9: Compute d_R1, d_R2 --------
    // Load t1_x and t2_x and compute d_R_rel
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER9_d_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
    }
    __syncthreads();
    copyContiguousMemory(reinterpret_cast<const char *>(t1xGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER9_T1X),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    if (threadIdxInBlock < minNumThreadsInBlock) {
      for (int j = 0; j < 3; j++) {
        LAYER9_d_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER9_d_ESSENTIAL[pairIdxInBatch][rowIdxInBatch][j] *
            LAYER9_T1X[pairIdxInBatch][colIdxInBatch]
                      [j]; // Note that t1_x is transposed
      }
    }
    __syncthreads();
    copyContiguousMemory(reinterpret_cast<const char *>(t2xGlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER9_T2X),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    if (threadIdxInBlock < minNumThreadsInBlock) {
      for (int i = 0; i < 3; i++) {
        LAYER9_d_R_REL[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] -=
            LAYER9_T2X[pairIdxInBatch][i][rowIdxInBatch] *
            LAYER9_d_ESSENTIAL[pairIdxInBatch][i]
                              [colIdxInBatch]; // Note that t2_x is transposed
      }
    }
    __syncthreads();
    // Load R2 and compute d_R1
    copyContiguousMemory(reinterpret_cast<const char *>(R2GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER9_R2),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER9_d_R1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        LAYER9_d_R1[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER9_d_R_REL[pairIdxInBatch][i][rowIdxInBatch] *
            LAYER9_R2[pairIdxInBatch][i]
                     [colIdxInBatch]; // Note that R_rel is transposed
      }
    }
    __syncthreads();
    // Write d_R1 to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER9_d_R1),
                         reinterpret_cast<char *>(dR1GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    // Load R1 and compute d_R2
    copyContiguousMemory(reinterpret_cast<const char *>(R1GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         reinterpret_cast<char *>(LAYER9_R1),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
    if (threadIdxInBlock < minNumThreadsInBlock) {
      LAYER9_d_R2[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] = 0;
      for (int i = 0; i < 3; i++) {
        LAYER9_d_R2[pairIdxInBatch][rowIdxInBatch][colIdxInBatch] +=
            LAYER9_d_R_REL[pairIdxInBatch][rowIdxInBatch][i] *
            LAYER9_R1[pairIdxInBatch][i][colIdxInBatch];
      }
    }
    __syncthreads();
    // Write d_R2 to global memory
    copyContiguousMemory(reinterpret_cast<const char *>(LAYER9_d_R2),
                         reinterpret_cast<char *>(dR2GlobalPtr) +
                             startImagePairIdx * 9 * sizeof(T),
                         numImagePairsInBatch * sizeof(T) * 9);
    __syncthreads();
  }
}

// Thin C++ wrapper that launches the kernel on the current stream
template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
epipolar_gradient(const at::Tensor &R1, const at::Tensor &R2,
                  const at::Tensor &t1, const at::Tensor &t2,
                  const at::Tensor &f1Inv, const at::Tensor &f2Inv,
                  const at::Tensor &W, at::Tensor &loss, at::Tensor &dR1,
                  at::Tensor &dR2, at::Tensor &dt1, at::Tensor &dt2,
                  at::Tensor &dF1Inv, at::Tensor &dF2Inv,
                  at::Tensor &bufferRrel, at::Tensor &buffert1x,
                  at::Tensor &buffert2x, at::Tensor &bufferEssential,
                  at::Tensor &bufferFundamental) {
  TORCH_CHECK(R1.is_cuda() && R2.is_cuda() && t1.is_cuda() && t2.is_cuda() &&
                  W.is_cuda(),
              "Tensors must be on CUDA");
  int numImagePairs = R1.size(0);

  constexpr int numThreadsPerBlock = BATCH_SIZE * 3 * 3;
  const int numBlocks = 1024;

  // Set loss tensor to zero
  loss.zero_();

  // Launch the kernel
  epipolarKernel<T>
      <<<numBlocks, numThreadsPerBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
          R1.data_ptr<T>(), R2.data_ptr<T>(), t1.data_ptr<T>(),
          t2.data_ptr<T>(), f1Inv.data_ptr<T>(), f2Inv.data_ptr<T>(),
          W.data_ptr<T>(), bufferRrel.data_ptr<T>(), buffert1x.data_ptr<T>(),
          buffert2x.data_ptr<T>(), bufferEssential.data_ptr<T>(),
          bufferFundamental.data_ptr<T>(), loss.data_ptr<T>(),
          dR1.data_ptr<T>(), dR2.data_ptr<T>(), dt1.data_ptr<T>(),
          dt2.data_ptr<T>(), dF1Inv.data_ptr<T>(), dF2Inv.data_ptr<T>(),
          numImagePairs);

  // Synchronize to ensure kernel execution is complete
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error in epipolar_gradient kernel: " +
                             std::string(cudaGetErrorString(err)));
  }
  cudaDeviceSynchronize();

  return {loss, dR1, dR2, dt1, dt2, dF1Inv, dF2Inv};
}

// Explicit instantiation
template std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
                    at::Tensor, at::Tensor>
epipolar_gradient<float>(const at::Tensor &R1, const at::Tensor &R2,
                         const at::Tensor &t1, const at::Tensor &t2,
                         const at::Tensor &f1Inv, const at::Tensor &f2Inv,
                         const at::Tensor &W, at::Tensor &loss, at::Tensor &dR1,
                         at::Tensor &dR2, at::Tensor &dt1, at::Tensor &dt2,
                         at::Tensor &dF1Inv, at::Tensor &dF2Inv,
                         at::Tensor &bufferRrel, at::Tensor &buffert1x,
                         at::Tensor &buffert2x, at::Tensor &bufferEssential,
                         at::Tensor &bufferFundamental);

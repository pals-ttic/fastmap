#include "kernels.h"

constexpr int WARP_SIZE = 32;
constexpr int BATCH_SIZE = 8; // Adjust as needed
static_assert(BATCH_SIZE <= WARP_SIZE,
              "BATCH_SIZE must be less than or equal to WARP_SIZE");

__device__ __forceinline__ void
copyGlobalMemoryToShared(const char *__restrict__ globalPtr,
                         char *__restrict__ sharedPtr, int numBytes) {
  const int threadIdxInBlock = threadIdx.x;
  const int numThreadsInBlock = blockDim.x;
  for (int i = threadIdxInBlock; i < numBytes; i += numThreadsInBlock) {
    sharedPtr[i] = globalPtr[i];
  }
}

template <typename T>
__global__ void epipolarAdjustmentKernel(
    const int64_t *__restrict__ imageCameraIndices, const T *__restrict__ Rw2c,
    const T *__restrict__ tw2c, const T *__restrict__ invFocalScale,
    const T *__restrict__ W, T *__restrict__ out, int numImagePairs) {

  const int threadIdxInBlock = threadIdx.x;
  const int numThreadsInBlock = blockDim.x;
  const int blockStride = gridDim.x * BATCH_SIZE;

  // Allocate shared memory for the batch
  __shared__ int64_t sharedImageCameraIndices[BATCH_SIZE * 4];
  __shared__ T sharedRw2c1[BATCH_SIZE][3][3]; // 3x3 matrix per batch
  __shared__ T sharedRw2c2[BATCH_SIZE][3][3]; // 3x3 matrix per batch
  __shared__ T sharedTw2c1[BATCH_SIZE][3];    // 3D vector per batch
  __shared__ T sharedTw2c2[BATCH_SIZE][3];    // 3D vector per batch
  __shared__ T sharedInvFocalScale1[BATCH_SIZE];
  __shared__ T sharedInvFocalScale2[BATCH_SIZE];
  __shared__ T sharedRRel[BATCH_SIZE][3][3]; // 3x3 matrix per batch

  // Loop until all batches for the block are processed
  for (int blockStartIdx = blockIdx.x * BATCH_SIZE;
       blockStartIdx < numImagePairs; blockStartIdx += blockStride) {
    int numImagePairsInBlock = BATCH_SIZE;
    if (blockStartIdx + BATCH_SIZE > numImagePairs) {
      numImagePairsInBlock = numImagePairs - blockStartIdx;
    }
    copyGlobalMemoryToShared(
        reinterpret_cast<const char *>(imageCameraIndices + blockStartIdx * 4),
        reinterpret_cast<char *>(sharedImageCameraIndices),
        sizeof(int64_t) * (numImagePairsInBlock * 4));
  }

  // Explicit instantiations
  template __global__ void epipolarAdjustmentKernel<float>(
      const int64_t *, const float *, const float *, const float *,
      const float *, float *, int64_t);

  // Thin C++ wrapper that launches the kernel on the current stream
  at::Tensor epipolar_adjustment_compute_gradient(const at::Tensor &A,
                                                  const at::Tensor &B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.sizes() == B.sizes(), "Input sizes must match");
    auto C = at::empty_like(A);

    const int64_t N = A.numel();
    constexpr int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Dispatch on scalar type (float, double, half, …)
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        A.scalar_type(), "epipolar_adjustment", [&] {
          epipolarAdjustmentKernel<scalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
                  C.data_ptr<scalar_t>(), N);
        });

    return C;
  }

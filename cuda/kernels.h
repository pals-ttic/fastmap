#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>   // (same, either header is fine)
#include <cuda_runtime.h>
#include <torch/extension.h>

at::Tensor epipolar_gradient(const at::Tensor &R1, const at::Tensor &R2,
                             const at::Tensor &t1, const at::Tensor &t2,
                             const at::Tensor &W);

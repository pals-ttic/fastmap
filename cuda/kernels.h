#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>   // (same, either header is fine)
#include <cuda_runtime.h>
#include <torch/extension.h>

at::Tensor epipolar_adjustment_compute_gradient(const at::Tensor &A,
                                                const at::Tensor &B);

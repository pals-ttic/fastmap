#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>   // (same, either header is fine)
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
epipolar_gradient(const at::Tensor &R1, const at::Tensor &R2,
                  const at::Tensor &t1, const at::Tensor &t2,
                  const at::Tensor &f1Inv, const at::Tensor &f2Inv,
                  const at::Tensor &W);

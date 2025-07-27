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
                  const at::Tensor &W, at::Tensor &loss, at::Tensor &dR1,
                  at::Tensor &dR2, at::Tensor &dt1, at::Tensor &dt2,
                  at::Tensor &dF1Inv, at::Tensor &dF2Inv,
                  at::Tensor &bufferRrel, at::Tensor &buffert1x,
                  at::Tensor &buffert2x, at::Tensor &bufferEssential,
                  at::Tensor &bufferFundamental);

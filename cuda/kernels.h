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

template <typename T>
void rotation_gradient(const at::Tensor &Rrel, const at::Tensor &Rw2c1,
                       const at::Tensor &Rw2c2, at::Tensor &loss,
                       at::Tensor &dRw2c1, at::Tensor &dRw2c2, T clampThr);

template <typename T>
void translation_gradient(const at::Tensor &o1, const at::Tensor &o2,
                          const at::Tensor &o12GT, at::Tensor &loss,
                          at::Tensor &do1, at::Tensor &do2);

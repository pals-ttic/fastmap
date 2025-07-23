#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA extension for FastMap";
  m.def("epipolar_gradient", &epipolar_gradient,
        "Compute the epipolar gradient");
}

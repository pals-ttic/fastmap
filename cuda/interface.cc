#include <c10/cuda/CUDAStream.h> // (same, either header is fine)
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA extension for epipolar adjustment";
  m.def("epipolar_adjustment_compute_gradient",
        &epipolar_adjustment_compute_gradient,
        "Add two same-shape CUDA tensors");
}

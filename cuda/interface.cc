#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA extension for FastMap";
  m.def("epipolar_gradient", &epipolar_gradient<float>, py::arg("R1"),
        py::arg("R2"), py::arg("t1"), py::arg("t2"), py::arg("f1_inv"),
        py::arg("f2_inv"), py::arg("W"), py::arg("loss"), py::arg("d_R1"),
        py::arg("d_R2"), py::arg("d_t1"), py::arg("d_t2"), py::arg("d_f1_inv"),
        py::arg("d_f2_inv"), py::arg("buffer_R_rel"), py::arg("buffer_t1_x"),
        py::arg("buffer_t2_x"), py::arg("buffer_essential"),
        py::arg("buffer_fundamental"), "Compute the epipolar gradient");
}

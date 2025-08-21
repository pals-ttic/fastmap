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
  m.def("rotation_gradient", &rotation_gradient<float>, py::arg("R_rel"),
        py::arg("R_w2c1"), py::arg("R_w2c2"), py::arg("loss"),
        py::arg("d_R_w2c1"), py::arg("d_R_w2c2"), py::arg("clamp_thr"),
        "Compute the rotation gradient");
  m.def("translation_gradient", &translation_gradient<float>, py::arg("o1"),
        py::arg("o2"), py::arg("o12_gt"), py::arg("loss"), py::arg("d_o1"),
        py::arg("d_o2"), "Compute the translation gradient");
}
void translation_gradient(const at::Tensor &o1, const at::Tensor &o2,
                          const at::Tensor &o12GT, at::Tensor &loss,
                          at::Tensor &do1, at::Tensor &do2);

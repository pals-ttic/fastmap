#include <ATen/cuda/CUDAContext.h> // for at::cuda::getCurrentCUDAStream()
#include <c10/cuda/CUDAStream.h>   // (same, either header is fine)
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void vector_add_kernel(const T *__restrict__ A,
                                  const T *__restrict__ B, T *__restrict__ C,
                                  int64_t N) {
  // Global thread index
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// Explicit instantiations so the symbols exist in the fat-bin.
// (float and double are enough for most demos)
template __global__ void vector_add_kernel<float>(const float *, const float *,
                                                  float *, int64_t);
template __global__ void
vector_add_kernel<double>(const double *, const double *, double *, int64_t);

// Thin C++ wrapper that launches the kernel on the current stream
at::Tensor vector_add_cuda(const at::Tensor &A, const at::Tensor &B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(A.sizes() == B.sizes(), "Input sizes must match");
  auto C = at::empty_like(A);

  const int64_t N = A.numel();
  constexpr int threads = 256;
  const int blocks = (N + threads - 1) / threads;

  // Dispatch on scalar type (float, double, half, …)
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "vector_add_cuda", [&] {
    vector_add_kernel<scalar_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            A.data_ptr<scalar_t>(), B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(), N);
  });

  return C;
}

// Python binding
// ----------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Simple CUDA vector add (A + B)";
  m.def("add", &vector_add_cuda, "Add two same-shape CUDA tensors");
}

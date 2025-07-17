from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME='fastmap'

this_dir = Path(__file__).parent

setup(
    name=PACKAGE_NAME,
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE_NAME}.vector_add_ext",
            sources=[
                str(this_dir / "cuda" / "epipolar_adjustment.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    # Uncomment & tweak for specific GPU arch if desired:
                    # "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)

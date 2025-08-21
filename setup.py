import subprocess
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = "fastmap"
THIS_DIR = Path(__file__).parent
SOURCE_DIR = THIS_DIR / "cuda"


class BuildExtWithCompDb(BuildExtension):
    def run(self):
        super().run()  # regular build
        build_dir = Path(self.build_temp)  # e.g. build/temp.linux‑…
        ninja_file = build_dir / "build.ninja"
        if ninja_file.exists():
            compdb = subprocess.check_output(
                ["ninja", "-C", str(build_dir), "-t", "compdb"]
            )
            (Path.cwd() / "compile_commands.json").write_bytes(compdb)


setup(
    name=PACKAGE_NAME,
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name=f"{PACKAGE_NAME}.cuda",
            sources=[
                str(SOURCE_DIR / "rotation.cu"),
                str(SOURCE_DIR / "translation.cu"),
                str(SOURCE_DIR / "epipolar.cu"),
                str(SOURCE_DIR / "interface.cc"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    # Uncomment & tweak for specific GPU arch if desired:
                    # "-gencode=arch=compute_80,code=sm_80",
                ],
                # "nvcc": ["-O0", "-g", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtWithCompDb.with_options(no_python_abi_suffix=True)},
)

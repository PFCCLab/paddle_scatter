import os
import re
import os.path as osp

import paddle
from paddle.utils.cpp_extension import CppExtension
from paddle.utils.cpp_extension import CUDAExtension
from paddle.utils.cpp_extension import setup


def get_version():
    current_dir = osp.dirname(osp.abspath(__file__))
    with open(osp.join(current_dir, "paddle_scatter/__init__.py")) as f:
        content = f.read()
    version_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Cannot find __version__ in paddle_scatter/__init__.py")

__version__ = get_version()


def set_cuda_archs():
    cuda_major, _ = paddle.version.cuda_version.split(".")

    if int(paddle.version.major) >= 3 and int(paddle.version.minor) >= 3:
        if int(cuda_major) >= 13:
            paddle_known_gpu_archs = ['Turing','Ampere','Ada','Hopper','Blackwell']
        elif int(cuda_major) >= 12:
            paddle_known_gpu_archs = ['Pascal','Volta','Turing','Ampere','Ada','Hopper']
        elif int(cuda_major) >= 11:
            paddle_known_gpu_archs = ['Pascal','Volta','Turing','Ampere','Ada']
        elif int(cuda_major) >= 10:
            paddle_known_gpu_archs = ['Pascal','Volta','Turing']
        else:
            raise ValueError("Not support cuda version.")
    else:
        # compatible with paddle < 3.3.0
        paddle_known_gpu_archs = paddle.version.compiled_cuda_archs

    os.environ["PADDLE_CUDA_ARCH_LIST"] = " ".join(
        [str(arch) for arch in paddle_known_gpu_archs]
    )


def get_sources():
    csrc_dir_path = os.path.join(os.path.dirname(__file__), "csrc")
    cpp_files = []
    for item in os.listdir(csrc_dir_path):
        if paddle.device.is_compiled_with_cuda():
            if item.endswith(".cc") or item.endswith(".cu"):
                cpp_files.append(os.path.join(csrc_dir_path, item))
        else:
            if item.endswith(".cc"):
                cpp_files.append(os.path.join(csrc_dir_path, item))
    return csrc_dir_path, cpp_files


def get_extensions():
    Extension = CppExtension
    extra_compile_args = {'cxx': ['-O3']}
    if paddle.device.is_compiled_with_cuda():
        set_cuda_archs()
        Extension = CUDAExtension
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3"]
        nvcc_flags += ["--expt-relaxed-constexpr"]
        extra_compile_args["nvcc"] = nvcc_flags

    src = get_sources()
    ext_modules = [
        Extension(
            sources=src[1],
            include_dirs=src[0],
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


if __name__ == "__main__":
    setup(
        name="paddle_scatter_ops",
        version=__version__,
        author="NKNaN",
        url="https://github.com/PFCCLab/paddle_scatter",
        description="Paddle extension of scatter and segment operators with min and max reduction methods, originally from https://github.com/rusty1s/pytorch_scatter",
        ext_modules=get_extensions(),
    )

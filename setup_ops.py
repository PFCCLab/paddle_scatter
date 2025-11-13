import os
import os.path as osp
import re

import paddle
from paddle.utils.cpp_extension import CppExtension
from paddle.utils.cpp_extension import CUDAExtension
from paddle.utils.cpp_extension import setup

PADDLE_PATH = os.path.dirname(paddle.__file__)
PADDLE_INCLUDE_PATH = os.path.join(PADDLE_PATH, "include")
PADDLE_LIB_PATH = os.path.join(PADDLE_PATH, "libs")
BASE_DIR = "/workspace/wangguan12/xpu"
os.environ["XHPC_PATH"] = BASE_DIR + "/xhpc-ubuntu2004_x86_64"
os.environ["XRE_PATH"] = BASE_DIR + "/xre-Linux-x86_64-5.0.21.22"
os.environ["CLANG_PATH"] = BASE_DIR + "/xtdk-llvm15-ubuntu2004_x86_64"
os.environ["BKCL_PATH"] = BASE_DIR + "/xccl_rdma-ubuntu_x86_64"
# os.environ['XFT_PATH'] = os.environ['XHPC_PATH']  # XFT在XHPC目录下
# os.environ['XBLAS_PATH'] = os.environ['XHPC_PATH']  # XBLAS在XHPC目录下

BKCL_PATH = os.getenv("BKCL_PATH")
if BKCL_PATH is None:
    BKCL_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xpu")
    BKCL_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libbkcl.so")
else:
    BKCL_INC_PATH = os.path.join(BKCL_PATH, "include")
    BKCL_LIB_PATH = os.path.join(BKCL_PATH, "so", "libbkcl.so")

# XFT_PATH = os.getenv("XFT_PATH")
# if XFT_PATH is None:
#     XFT_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xft")
#     XFT_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxft.so")
# else:
#     XFT_INC_PATH = os.path.join(XFT_PATH, "include")
#     XFT_LIB_PATH = os.path.join(XFT_PATH, "so", "libxft.so")

XRE_PATH = os.getenv("XRE_PATH")
if XRE_PATH is None:
    XRE_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xre")
    XRE_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxpucuda.so")
else:
    XRE_INC_PATH = os.path.join(XRE_PATH, "include")
    XRE_LIB_PATH = os.path.join(XRE_PATH, "so", "libxpucuda.so")

# XFA_PATH = os.getenv("XFA_PATH")
# if XFA_PATH is None:
#     XFA_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xhpc", "xfa")
#     XFA_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxpu_flash_attention.so")
# else:
#     XFA_INC_PATH = os.path.join(XFA_PATH, "include")
#     XFA_LIB_PATH = os.path.join(XFA_PATH, "so", "libxpu_flash_attention.so")

# XBLAS_PATH = os.getenv("XBLAS_PATH")
# if XBLAS_PATH is None:
#     XBLAS_INC_PATH = os.path.join(PADDLE_INCLUDE_PATH, "xhpc", "xblas")
#     XBLAS_LIB_PATH = os.path.join(PADDLE_LIB_PATH, "libxpu_blas.so")
# else:
#     XBLAS_INC_PATH = os.path.join(XBLAS_PATH, "include")
#     XBLAS_LIB_PATH = os.path.join(XBLAS_PATH, "so", "libxpu_blas.so")


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
    major, _ = paddle.version.cuda_version.split(".")
    if int(major) >= 12:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80, 90]
    elif int(major) >= 11:
        paddle_known_gpu_archs = [50, 60, 61, 70, 75, 80]
    elif int(major) >= 10:
        paddle_known_gpu_archs = [50, 52, 60, 61, 70, 75]
    else:
        raise ValueError("Not support cuda version.")

    os.environ["PADDLE_CUDA_ARCH_LIST"] = ",".join(
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
    return [csrc_dir_path], cpp_files


def get_extensions():
    Extension = CppExtension
    extra_objects = []
    include_dirs, sources = get_sources()

    extra_compile_args = {"cxx": ["-O3"]}
    if paddle.device.is_compiled_with_cuda():
        set_cuda_archs()
        Extension = CUDAExtension
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O3"]
        nvcc_flags += ["--expt-relaxed-constexpr"]
        extra_compile_args["nvcc"] = nvcc_flags
    elif paddle.device.is_compiled_with_xpu():
        include_dirs += [
            XRE_INC_PATH,
            # XFT_INC_PATH,
            BKCL_LIB_PATH,
            # XFA_INC_PATH,
            # XBLAS_INC_PATH,
        ]
        extra_objects += [
            XRE_LIB_PATH,
            # XFT_LIB_PATH,
            BKCL_LIB_PATH,
            # XFA_LIB_PATH,
            # XBLAS_LIB_PATH,
        ]
        extra_compile_args["cxx"] = ["-D_GLIBCXX_USE_CXX11_ABI=1", "-DPADDLE_WITH_XPU"]
    else:
        raise ("Only CUDA and XPU devices are supported")

    ext_modules = [
        Extension(
            sources=sources,
            include_dirs=include_dirs,
            extra_objects=extra_objects,
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
        description="Paddle extension of scatter and segment operators \
            with min and max reduction methods, \
            originally from https://github.com/rusty1s/pytorch_scatter",
        ext_modules=get_extensions(),
    )

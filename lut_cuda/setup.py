from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

nvcc_flags = [
    '-O3',
    '-use_fast_math',
    '--ptxas-options=-v',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__'
]


setup(
    name='lut_cuda_grouped',
    ext_modules=[
        CUDAExtension(
            name='lut_cuda_grouped',
            sources=['lut_cuda_backend.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='lut_cuda_grouped_binary',
    ext_modules=[
        CUDAExtension(
            name='lut_cuda_grouped_binary',
            sources=['lut_cuda_backend_binary.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='lut_conv_fp32_cuda', # 这是你在 Python 中 import 的名字
    ext_modules=[
        CUDAExtension(
            name='lut_conv_fp32_cuda',
            sources=[
                'lut_conv_cuda_backend.cu', # 你的 CUDA 源码文件
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
setup(
    name='lut_conv_binary_cuda',
    ext_modules=[
        CUDAExtension(
            name='lut_conv_binary_cuda',
            sources=['lut_conv_binary_cuda_backend.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='lut_conv_bffb_cuda',
    ext_modules=[
        CUDAExtension(
            name='lut_conv_bffb_cuda',
            sources=['lut_conv_bffb_cuda_backend.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-gencode=arch=compute_80,code=sm_80', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__'
]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

setup(
    name='lut_conv_bafw_cuda',
    ext_modules=[
        CUDAExtension(
            name='lut_conv_bafw_cuda',
            sources=['lut_conv_bafw_cuda_backend.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math', '-gencode=arch=compute_80,code=sm_80', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__'
]}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

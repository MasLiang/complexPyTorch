from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lut_conv_cuda',
    version='1.0',
    description='Hardware-Aware LUT Convolution CUDA Backends (LUT4, LUT5, LUT6)',
    ext_modules=[
        # 1. 浮点多线性插值后端 (用于 Phase 3 / 联合优化)
        CUDAExtension(
            name='lut_cuda_floating', # 编译后在 Python 中 import lut_cuda_floating
            sources=['lut_conv_cuda_backend.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}
        ),
        # 2. 二值极速后端 (用于 Phase 4 / 极限部署)
        CUDAExtension(
            name='lut_cuda_binary',   # 编译后在 Python 中 import lut_cuda_binary
            sources=['lut_conv_binary_cuda_backend.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
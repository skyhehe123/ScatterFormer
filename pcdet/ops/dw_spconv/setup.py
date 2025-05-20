import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUDA version
cuda_version = os.environ.get("CUDA_VERSION", "")
if not cuda_version:
    try:
        import torch
        cuda_version = torch.version.cuda
    except:
        raise RuntimeError("CUDA version not found. Please set CUDA_VERSION environment variable.")

# Define the CUDA extension
ext_modules = [
    CUDAExtension(
        name='dw_spconv',
        sources=['dw_spconv.cu'],
        extra_compile_args={
            'cxx': ['-O2'],
            'nvcc': [
                '-O2',
                '-arch=sm_60',  # Minimum compute capability
                '-gencode', f'arch=compute_60,code=sm_60',
                '-gencode', f'arch=compute_70,code=sm_70',
                '-gencode', f'arch=compute_75,code=sm_75',
                '-gencode', f'arch=compute_80,code=sm_80',
                '-gencode', f'arch=compute_86,code=sm_86',
            ]
        }
    )
]

setup(
    name='spdwconv',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
) 
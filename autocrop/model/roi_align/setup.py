import os
import torch
from pkg_resources import parse_version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

min_version = parse_version('1.0.0')
current_version = parse_version(torch.__version__)

if current_version < min_version:  # PyTorch before 1.0
    raise NotImplementedError('Only support torch>=1.0.0')

current_dir = os.path.dirname(os.path.realpath(__file__))
if torch.cuda.is_available():
    setup(
        name='roi_align_api',
        ext_modules=[
            CUDAExtension(
                name='roi_align_api',
                sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu'],
                include_dirs=[current_dir] + torch.utils.cpp_extension.include_paths(cuda=True)
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    setup(
        name='roi_align_api',
        ext_modules=[
            CppExtension(
                name='roi_align_api',
                sources=['src/roi_align.cpp'],
                include_dirs=[current_dir] + torch.utils.cpp_extension.include_paths(cuda=False)
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        })

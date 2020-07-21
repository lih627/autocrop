import setuptools
import os
from os.path import join as pjoin


def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


NEED_PACKAGES = ['torch']

for package in NEED_PACKAGES:
    install_and_import('torch')


def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            return None

    return cudaconfig


current_dir = os.path.dirname(os.path.realpath(__file__))


def generate_ext():
    import torch
    from torch.utils.cpp_extension import CppExtension, CUDAExtension
    rod_src_path = os.path.join(current_dir, 'autocrop/model/rod_align/src/')
    roi_src_path = os.path.join(current_dir, 'autocrop/model/roi_align/src/')
    if torch.cuda.is_available():
        roi_align = CUDAExtension(
            name='roi_align_api',
            sources=['autocrop/model/roi_align/src/roi_align_cuda.cpp',
                     'autocrop/model/roi_align/src/roi_align_kernel.cu'],
            include_dirs=[current_dir, rod_src_path, roi_src_path] +
                         torch.utils.cpp_extension.include_paths(cuda=True)
        )
        rod_align = CUDAExtension(
            name='rod_align_api',
            sources=['autocrop/model/rod_align/src/rod_align_cuda.cpp',
                     'autocrop/model/rod_align/src/rod_align_kernel.cu'],
            include_dirs=[current_dir, rod_src_path, roi_src_path]
                         + torch.utils.cpp_extension.include_paths(cuda=True)
        )
    else:
        roi_align = CppExtension(name='roi_align_api',
                                 sources=['autocrop/model/roi_align/src/roi_align.cpp'],
                                 include_dirs=[current_dir, rod_src_path, roi_src_path]
                                              + torch.utils.cpp_extension.include_paths(cuda=False)
                                 )
        rod_align = CppExtension(name='rod_align_api',
                                 sources=['autocrop/model/rod_align/src/rod_align.cpp'],
                                 include_dirs=[current_dir, rod_src_path, roi_src_path]
                                              + torch.utils.cpp_extension.include_paths(cuda=False)
                                 )

    ext_m = [roi_align, rod_align]
    return ext_m


setuptools.setup(
    name='auto_crop',
    author='Hao Li',
    author_email='lih627@outlook.com',
    version='0.2.1',
    description='Smart auto cropping tool that supports any aspect ratio',
    long_description=''.join(open('README.md', 'r').readlines()),
    long_description_content_type='text/markdown',
    url='https://github.com/lih627/autocrop',
    python_requires='>=3.6',
    license='MIT',
    ext_modules=generate_ext(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=[
        'torch>=1.1',
        'torchvision>=0.3.0',
        'numpy',
        'opencv-python',
        'face-detection>=0.1.4'
    ],
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages()
)

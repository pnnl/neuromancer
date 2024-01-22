from setuptools import setup
import torch.cuda
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []
if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'factor_multiply', [
            'factor_multiply.cpp',
            'factor_multiply_cuda.cu'
        ],
        extra_compile_args={'cxx': ['-g', '-march=native'],
                            # 'nvcc': ['-arch=sm_60', '-O2', '-lineinfo']})
                            'nvcc': ['-O2', '-lineinfo']})
    ext_modules.append(extension)
# extension = CppExtension('factor_multiply', ['factor_multiply.cpp'], extra_compile_args=['-march=native'])
# extension = CppExtension('factor_multiply', ['factor_multiply.cpp'])
# ext_modules.append(extension)

setup(
    name='extension',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})

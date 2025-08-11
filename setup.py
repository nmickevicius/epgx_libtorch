from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Ensure you point to the correct LibTorch path
libtorch_path = os.getenv('LIBTORCH_PATH', '/home/nikolai/dev/libtorch')

setup(
    name='epgx_mt_gre',
    ext_modules=[
        CUDAExtension(
            'epgx_mt_gre',
            ['epgx_mt.cpp'],
            include_dirs=[
                os.path.join(libtorch_path, 'include'),
                os.path.join(libtorch_path, 'include', 'torch', 'csrc', 'api', 'include')
            ],
            library_dirs=[os.path.join(libtorch_path, 'lib')],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2']
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
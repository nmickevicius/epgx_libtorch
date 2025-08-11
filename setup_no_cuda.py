from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='epgx_mt',
    ext_modules=[
        CppExtension(
            name='epgx_mt',
            sources=['epgx_mt.cpp'],
            extra_compile_args={"cxx": ["-std=c++17", "-mmacosx-version-min=10.15"]},
            extra_link_args=["-mmacosx-version-min=10.15"],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
from setuptools import setup, Extension
import pybind11

setup(
    name='pyevo',
    version='0.1',
    ext_modules=[
        Extension(
            'pyevo',
            ['pyevo.cpp'],
            include_dirs=[
                pybind11.get_include(),
                '.',  # Current directory for headers
                '../../evo'  # Path to evo headers
            ],
            library_dirs=['../../build'],
            libraries=['evo', 'm'],
            extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++11', '-fPIC']
        )
    ],
)

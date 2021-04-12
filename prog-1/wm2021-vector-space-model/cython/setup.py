from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

setup(
    ext_modules = cythonize("model.pyx", language_level = "3"),
    include_dirs=[numpy.get_include()]
)
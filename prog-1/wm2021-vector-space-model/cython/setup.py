from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

os.environ['CFLAGS'] = '-O3 -Wall -std=c++11'

setup(
    ext_modules = cythonize("model.pyx", language_level = "3"),
    include_dirs=[numpy.get_include()]
)
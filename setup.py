import numpy as np
import glob
#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import setup, Extension
from Cython.Distutils import build_ext

NAME = "edpyt"
VERSION = "0.1"
DESCR = "A small template project for exact diagonalization."
URL = "https://gitlab.ethz.ch/ggandus/edpyt.git"

AUTHOR = "Guido Gandus"
EMAIL = "ggandus@ethz.ch"

LICENSE = "Apache 2.0"

SRC_DIR = "edpyt"
PACKAGES = [SRC_DIR]


ext_1 = Extension(SRC_DIR + '._psparse',
    [SRC_DIR + '/psparse.pyx', SRC_DIR + '/cs_gaxpy.c'],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
    include_dirs = [np.get_include(),'edpyt'],
    extra_link_args=['-fopenmp'])

ext_2 = Extension(SRC_DIR + '._continued_fraction',
    [SRC_DIR + '/continued_fraction.pyx'],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math'],
    include_dirs = [np.get_include(),'edpyt'],
    extra_link_args=['-fopenmp'])

EXTENSIONS = [ext_1,ext_2]

requires = {
    "python_requires": ">= " + "3.7",
    "install_requires": [
        "setuptools",
        "cython",
        "mpmath",
        "numpy >= " + "1.19" + ", < 1.22",
        "scipy >= " + "1.6",
        "numba >= " + "0.51",
        "ase",
    ]}

setup(packages=PACKAGES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=EXTENSIONS,
      **requires
)

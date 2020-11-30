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
REQUIRES = ['numpy', 'cython', 'scipy']

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

EXTENSIONS = [ext_1]

setup(install_requires=REQUIRES,
      packages=PACKAGES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      url=URL,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=EXTENSIONS
)

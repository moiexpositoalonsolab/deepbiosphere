import sys
from setuptools import setup
import setuptools
# need the pkg_resources and python_requires to catch older
# versions of python, setuptools, and pip that don't use PEP 517
# or newer (aka not checking setup.cfg and pyproject.toml)
if sys.version_info < (3,7):
    sys.exit('Deepbiosphere only supports Python 3.7+')
# bare minimum setup.py so can still editable install packages
setup(python_requires='>=3.7', packages=setuptools.find_packages(where="./src"), package_dir={"":"src"})

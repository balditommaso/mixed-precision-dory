from setuptools import setup, Extension
import os


setup(name='dory',
      version='0.1',
      description='A library to deploy networks on MCUs',
      url='https://github.com/balditommaso/mixed-precision-dory/',
      author='Tommaso Baldi',
      author_email='tommaso.baldi@santannapisa.it',
      license='MIT',
      packages=setuptools.find_packages(),
	    python_requires='>=3.9',
	    install_requires=[
	        "onnx",
	        "numpy",
            "ortools",
            "mako"
	    ],
      package_data={"": ['Makefile*'], "": ['*.[json,c,h]']},
      include_package_data=True,
      zip_safe=False)
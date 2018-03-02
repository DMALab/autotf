#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='autotf',
      version='0.0.1',
      description='Automatic Machine Learning Toolkit for TensorFlow',
      author='AutoTF contributors',
      author_email='xx@pku.edu.cn',
      url='https://github.com/bluesjjw/AutoTF',
      license='BSD-3',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'six',
          'tensorflow'
      ],
      classifiers=[
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ],
      keywords=[
          'AutoTF',
          'TensorFlow',
          'Deep Learning',
          'Machine Learning',
          'Neural Networks',
          'AI'
      ]
      )

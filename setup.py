#!/usr/bin/env python

from setuptools import setup

setup(name='pysheds',
      version='0.2.6',
      description='🌎 Simple and fast watershed delineation in python.',
      author='Matt Bartos',
      author_email='mdbartos@umich.edu',
      url='http://open-storm.org',
      packages=["pysheds"],
      include_package_data = True,
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'pyproj',
          'scikit-image',
          'affine',
          'geojson',
          'rasterio>=1'
      ]
     )

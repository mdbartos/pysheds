#!/usr/bin/env python

from distutils.core import setup, find_packages

setup(name='pysheds',
      version='0.1',
      description='🌎 Simple and fast watershed delineation in python.',
      author='Matt Bartos',
      author_email='mdbartos@umich.edu',
      url='open-storm.org',
      packages=find_packages(exclude=["examples", "data", "tests"]),
     )

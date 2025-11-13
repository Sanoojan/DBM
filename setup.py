#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = "0.1.0"

subpackages = ["data", "dbm", "scripts", "stationary", "utils", "viz"]
setup(
    name="DBM",
    version=__version__,
    packages=find_packages(exclude=["test*", "doc*", *(f"{package}*" for package in subpackages)]),
    author="HAIL",
)

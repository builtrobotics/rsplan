# setup.py

import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "rs-path",
    version = "1.0.0",
    author = "Built Robotics",
    author_email = "tarakapoor9@gmail.com",
    description = ("Reeds-Shepp algorithm implementation in Python."),
    license = "MIT",
    keywords = "reeds-shepp path planning",
    url = "https://github.com/builtrobotics/rs",
    packages=['rs-path'],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)

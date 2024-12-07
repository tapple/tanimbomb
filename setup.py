#!/usr/bin/env python
from setuptools import setup
setup(name='tanimbomb',
    entry_points={'console_scripts': ['animDump = animDump:main']},
    py_modules=["animDump"],
    install_requires=['numpy'],
)

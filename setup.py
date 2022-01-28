#!/usr/bin/env python3
from setuptools import setup

# installation script, install using
# pip3 install -e .

setup(
	name='speaker_recognition',
	version='1.0.0',
	description='',
	author='Martin Androvich',
	url='https://github.com/martinandrovich/speaker_recognition',
	packages=['speaker_recognition'],
	install_requires=[
		'torch',
	]
)

#!/usr/bin/env python3
from setuptools import setup

# installation script, install using
# python -m pip install -e .

setup(
	name="speaker_recognition",
	version="1.0.0",
	description="",
	author="Martin Androvich",
	url="https://github.com/martinandrovich/speaker_recognition",
	packages=["speaker_recognition"],
	python_requires="==3.9.*",
	install_requires=[
		"numpy",
		"torch", "torchvision", "torchaudio", "torchinfo",
		"matplotlib",
		"dill",
		"PySoundFile ; platform_system=='Windows'",
		"sox ; platform_system=='Linux'",
	]
)

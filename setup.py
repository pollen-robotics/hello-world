from setuptools import setup, find_packages
from os import path
import io

here = path.abspath(path.dirname(__file__))

with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='hello_world',
    version='1.0.0',
    description='Idle behaviors for Reachy robot',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pollen-robotics/hello_world',
    author='Pollen-Robotics',
    author_email='contact@pollen-robotics.com',
    packages=find_packages(exclude=['tests']),
    python_requires='>=3.5',
    install_requires=[
        'reachy-sdk',
        'numpy',
        'playsound',
    ],
)

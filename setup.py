#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Multimodal deep learning for renal cancer prognosis",
    author="Sean Klein",
    author_email="seapat@github",
    url="https://github.com/seapat/renal-cancer-dl",  
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)

''' instrucitons
# install via setup.py
cd /project
pip install -e .
# call script
cd /project/src
python some_file.py --accelerator 'ddp' --gpus 8
'''
#!/bin/bash
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow tensorflow_gan jupyter jupyterlab imageio matplotlib numpy scipy
python3 -m pip install git+https://github.com/tensorflow/docs
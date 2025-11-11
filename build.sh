#!/bin/bash
# Install setuptools first to provide distutils compatibility
pip install --upgrade setuptools wheel
# Then install all other requirements
pip install -r requirements.txt


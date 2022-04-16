#!/bin/bash

nvcc --version
python3 /simulationScript.py data/crystal.cif data/miller_indices.txt data/pixelsize_angstrom.txt


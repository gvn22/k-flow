#!/usr/bin/env bash

mpirun -np 6 python3 join_output.py
mpirun -np 6 python3 plot_slices.py snapshots/snapshots_s1.h5
eog frames/

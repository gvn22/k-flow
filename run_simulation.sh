#!/usr/bin/env bash

./clean_output.sh
mpiexec -np 6 python3 k-flow.py

# k-flow
implements GQL equations for simulation of 2D flows with two-scale Kolmogorov forcing.

## Usage
Set parameters including GQL cutoff in input.yaml (will need pyyaml module installed)
Then type:

```terminal
mpirun -np X python3 k-flow.py

```
or

Set cores etc in script and type:
```terminal
./run_simulation.sh

```

## Contact
To contribute or report errors, email: matgvn@leeds.ac.uk
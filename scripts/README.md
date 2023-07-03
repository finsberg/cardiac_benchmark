# Scripts

This folder contains script for submitting the jobs on the [EX3 cluster](https://www.ex3.simula.no). However, the scripts should be fairly generic since they use SLURM. The only thing you probably need to change in order to run on a different clusters are the modules that you load in the beginning and the file names.

## Generate geometry

Before running any of the you should generate the geometry.
```
cardiac-benchmark  create-geometry geometry.h5
```
Note that this requires `fenics` and `gmsh` to be be installed. To get this you can e.g use the `fenics-gmsh` docker image that we have created here: https://github.com/scientificcomputing/packages/pkgs/container/fenics-gmsh


## Submit benchmark

Step 0 Case A
```
python run_benchmark.py step0-case-a
```

Step 0 Case B
```
python run_benchmark.py step0-case-b
```

Step 1
```
python run_benchmark.py step1
```

Step 2
```
python run_benchmark.py step1
```

Run all benchmarks
```
python run_benchmark.py all
```

Note that it is also possible to add a the flag `--dry-run` to just check which benchmarks that are run.

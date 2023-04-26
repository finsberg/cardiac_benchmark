# Cardiac benchmark

This is the contribution to the cardiac mechanics benchmark from Simula Research Laboratory

## Installation

### Conda

Create the conda environment using the `environment.yml` file
```
conda env create -f environment.yml
```
Activate the enviroment
```
conda activate cardiac-benchmark
```
and finally install the `cardiac-benchmark` package (from the root of the repository)
```
python3 -m pip install .
```

### Docker

### Note for M1 Mac
FEniCS is currently not available through conda for M1 mac (unless you use Rosetta 2). If you are using M1 mac then you can use the provided docker image.

Run the following command to start the container interactively and mount the current directory
```
docker run --rm -v $PWD:/home/shared -w /home/shared -it finsberg/cardiac-benchmark
```
This should spin up a container with everything installed. You will also find the [Dockerfile](docker/Dockerfile) used for creating this image in this repo.

#### Known issues
If you get the following error
```
Traceback (most recent call last):
  File "/home/shared/benchmark.py", line 7, in <module>
    from geometry import EllipsoidGeometry
  File "/home/shared/geometry.py", line 9, in <module>
    import gmsh
  File "/usr/local/lib/gmsh.py", line 53, in <module>
    lib = CDLL(libpath)
  File "/usr/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /lib/aarch64-linux-gnu/libgomp.so.1: cannot allocate memory in static TLS block
```
then you should do the following
```
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```


### For developers

Developers should also install the pre-commit hook

```
python -m pip install pre-commit
pre-commit install
```

## Running the benchmark

You can run the command line interface directly, e.g
```
cardiac-benchmark step1
```
To see all steps that you can run, do
```
cardiac-benchmark --help
```
and to see the specific options for a given step you can do (for e.g `step1`)
```
cardiac-bencmark step1 --help
```

You can also use the python API
```python
import cardiac_benchmark

cardiac_benchmark.benchmark.run()
```

## License

MIT

## Authors

- Henrik Finsberg (henriknf@simula.no)
- Joakim Sundnes (sundnes@simula.no)

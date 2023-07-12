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

Run the following command to start the container interactively and mount the current directory
```
docker run --rm -v $PWD:/home/shared -w /home/shared -it ghcr.io/finsberg/cardiac_benchmark:latest
```
This should spin up a container with everything installed. You will also find the [Dockerfile](docker/Dockerfile) used for creating this image in this repo.


#### Note for M1 Mac
FEniCS is currently not available through conda for M1 mac (unless you use Rosetta 2). If you are using M1 mac then you can use the provided docker image.


### For developers

Developers should also install the pre-commit hook

```
python -m pip install pre-commit
pre-commit install
```

## Running the benchmark

You can run the command line interface directly, e.g
```
cardiac-benchmark benchmark1-step1
```
To see all steps that you can run, do
```
cardiac-benchmark --help
```
and to see the specific options for a given step you can do (for e.g `step1`)
```
cardiac-benchmark benchmark1-step1 --help
```

You can also use the python API
```python
import cardiac_benchmark

cardiac_benchmark.benchmark1.run()
```
which by default will run benchmark 1 - step 1.


### Options

- Create geometry for benchmark 1 (save lv ellipsoidal geometry to `geometry.h5`). Note: requires `gmsh`
    ```
    cardiac-benchmark create-geometry geometry.h5
    ```
- Run benchmark 1 step 0 case A
    ```
    cardiac-benchmark benchmark1-step0 a
    ```
- Run benchmark 1 step 0 case B
    ```
    cardiac-benchmark benchmark1-step0 b
    ```
- Run benchmark 1 step 1
    ```
    cardiac-benchmark benchmark1-step1
    ```
- Run benchmark 1 step 2 case A
    ```
    cardiac-benchmark benchmark1-step2 a
    ```
- Run benchmark 1 step 2 case B
    ```
    cardiac-benchmark benchmark1-step2 b
    ```
- Run benchmark 1 step 2 case C
    ```
    cardiac-benchmark benchmark1-step2 c
    ```
- Download coarse data for benchmark 2 to a folder called `data_coarse`
    ```
    cardiac-benchmark download-data-benchmark2 coarse --outdir=data_coarse
    ```
- Download fine data for benchmark 2 to a folder called `data_fine`
    ```
    cardiac-benchmark download-data-benchmark2 fine --outdir=data_fine
    ```
- Run benchmark 2 with fine data
    ```
    cardiac-benchmark benchmark2 data_coarse
    ```


## License

MIT

## Authors

- Henrik Finsberg (henriknf@simula.no)
- Joakim Sundnes (sundnes@simula.no)

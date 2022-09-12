# Cardiac benchmark

This is the contribution to the cardiac mechanics benchmark from Simula Research Laboratory

## Installation

Create the conda environment using the `environment.yml` file
```
conda env create -f environment.yml
```
Activate the enviroment
```
conda activate cardiac-benchmark
```

### Note for M1 Mac
FEniCS is currently not available through conda for M1 mac. If you are using M1 mac then you can use the provided docker image.

Run the following command to start the container interactively and mount the current directory
```
docker run --rm -v $PWD:/home/shared -w /home/shared -it finsberg/cardiac-benchmark
```
This should spin up a container with everything installed. You will also find the [Dockerfile](Dockerfile) used for creating this image in this repo.


### For developers

Developers should also install the pre-commit hook

```
python -m pip install pre-commit
pre-commit install
```

## Running the benchmark

```
python benchmark.py
```

## License

MIT

## Authors

- Henrik Finsberg (henriknf@simula.no)
- Joakim Sundnes (sundnes@simula.no)
- Jonas van den Brink (jvbrink@simula.no)

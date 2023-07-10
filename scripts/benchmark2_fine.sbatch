#!/bin/bash
# Basic usage:
#
# $ sbatch hello.sbatch
#
# The job is queued and starts as soon as resources are available. The
# script is then executed on one of the allocated tasks, and standard
# output and standard error streams will be redirected to files that
# are prefixed by the job ID and job name. Commands prefixed with
# `srun' are executed on every task acquired by the job allocation.
#
# The sbatch options below allocate a single task on a single node,
# using a single CPU core with a one-hour time limit. To override
# these defaults, you can also supply sbatch options on the command
# line. For example:
#
# $ sbatch --cpus-per-task=32 --time=02:00:00 hello.sbatch
#SBATCH --job-name="cardiac-benchmark"
#SBATCH --partition=defq
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=slurm-output/%j-%x-stdout.txt
#SBATCH --error=slurm-output/%j-%x-stderr.txt


module use /cm/shared/ex3-modules/0.6.1/modulefiles
module load python-fenics-dolfin-2019.1.0.post0

LIBRARY_ROOT=/home/henriknf/local/src/cardiac_benchmark

SCRATCH_DIRECTORY=/global/D1/homes/${USER}/cardiac_benchmark/benchmark2/fine/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
echo "Scratch directory: ${SCRATCH_DIRECTORY}"

echo "Run command: ${LIBRARY_ROOT}/venv/bin/cardiac-benchmark benchmark2 --outdir "${SCRATCH_DIRECTORY}" "${LIBRARY_ROOT}/data_fine""
# Assuming that you installed the software in a virutal environment in ${LIBRARY_ROOT}/venv
srun -n 8 ${LIBRARY_ROOT}/venv/bin/cardiac-benchmark benchmark2 --outdir "${SCRATCH_DIRECTORY}" "${LIBRARY_ROOT}/data_fine"
cp slurm-output/${SLURM_JOBID}-cardiac-benchmark-std* ${SCRATCH_DIRECTORY}
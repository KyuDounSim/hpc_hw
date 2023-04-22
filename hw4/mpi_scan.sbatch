#!/bin/bash

#SBATCH --job-name=mpi_scan
#SBATCH --output=mpi_scan.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00
#SBATCH --verbose

module purge 
module load openmpi/gcc/4.0.5 

mpirun -np 12 ./mpi_scan 1000 

## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


#!/bin/bash

#SBATCH --job-name=multi-pingpong
#SBATCH --output=multi-pingpong.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB 
#SBATCH --time=00:05:30 
#SBATCH --verbose

module purge 
module load openmpi/intel/4.0.5 

mpirun -np 2 ./pingpong 0 1

## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


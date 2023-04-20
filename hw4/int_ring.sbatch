#!/bin/bash

#SBATCH --job-name=int-ring
#SBATCH --output=int-ring.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00
#SBATCH --verbose

module purge 
module load openmpi/intel/4.0.5 

mpirun -np 3 ./int-ring 10

## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


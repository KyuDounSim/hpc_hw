#!/bin/bash

#SBATCH --job-name=pingpong
#SBATCH --output=pingpong.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB 
#SBATCH --time=00:05:30 
#SBATCH --verbose

module purge 
module load openmpi/gcc/4.0.5 

mpirun -np 4 ./pingpong 0 1 > pingpong_np4_0_1

## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


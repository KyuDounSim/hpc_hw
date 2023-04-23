#!/bin/bash

#SBATCH --job-name=int-ring
#SBATCH --output=int_ring.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00
#SBATCH --verbose

module purge 
module load openmpi/gcc/4.0.5 

mpirun -np 3 ./int_ring 10
mpirun -np 3 ./int_ring 100 
mpirun -np 3 ./int_ring 1000
mpirun -np 12 ./int_ring 10 
mpirun -np 12 ./int_ring 100
mpirun -np 12 ./int_ring 1000

## mpirun -np 3 ./int_ring 10 > int_3_ring_10.out

## mpirun -np 3 ./int_ring 100 > int_3_ring_100.out

## mpirun -np 3 ./int_ring 1000 > int_3_ring_1000.out

## mpirun -np 12 ./int_ring 10 > int_10_ring_10.out

## mpirun -np 12 ./int_ring 100 > int_10_ring_100.out

## mpirun -np 12 ./int_ring 1000 > int_10_ring_1000.out


## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


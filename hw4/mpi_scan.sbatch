#!/bin/bash

#SBATCH --job-name=mpi_scan
#SBATCH --output=mpi_scan.out
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=4
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00
#SBATCH --verbose

module purge 
module load openmpi/gcc/4.0.5 

##mpirun -np 2 ./mpi_scan 1000
##mpirun -np 5 ./mpi_scan 1000
##mpirun -np 10 ./mpi_scan 1000

mpirun -np 2 ./mpi_scan 48000
mpirun -np 5 ./mpi_scan 48000
mpirun -np 10 ./mpi_scan 48000
mpirun -np 20 ./mpi_scan 48000
 
mpirun -np 2 ./mpi_scan 4800000
mpirun -np 5 ./mpi_scan 4800000
mpirun -np 10 ./mpi_scan 4800000
mpirun -np 20 ./mpi_scan 4800000

## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


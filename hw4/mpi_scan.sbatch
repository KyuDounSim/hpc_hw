#!/bin/bash

#SBATCH --job-name=mpi_scan
#SBATCH --output=mpi_scan.out
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB 
#SBATCH --time=00:30:00
#SBATCH --verbose

module purge 
module load openmpi/gcc/4.0.5 

mpirun -np 1 ./mpi_scan 48000
mpirun -np 2 ./mpi_scan 48000
mpirun -np 4 ./mpi_scan 48000
mpirun -np 8 ./mpi_scan 48000
mpirun -np 20 ./mpi_scan 48000
 
mpirun -np 1 ./mpi_scan 4800000
mpirun -np 2 ./mpi_scan 4800000
mpirun -np 4 ./mpi_scan 4800000
mpirun -np 8 ./mpi_scan 4800000
mpirun -np 20 ./mpi_scan 4800000


## To execute this .sbatch file run the following,
## sbatch myscript.sbatch

## To view the status of the job,
## squeue -u $USER


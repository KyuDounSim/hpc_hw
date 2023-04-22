#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double execute_ring(long Nsize, long loop_number, MPI_Comm comm) {
  int rank, numprocs, namelen;
  //char processor_name[MPI_MAX_PROCESSOR_NAME];

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);
  //MPI_Get_processor_name(processor_name, &namelen);

  int* mssg = (int*) malloc(Nsize * sizeof(int));
  for (long idx = 0; idx < Nsize; ++idx) { mssg[idx] = 0; }

  MPI_Barrier(comm);
  // printf("Rank: %d, numprocs: %d, processor_name: %s\n", rank, numprocs, processor_name);
  double tt = MPI_Wtime();

  for(long repeat = 0; repeat < loop_number; ++repeat) {
    MPI_Status status;

    if(rank == 0) {
      MPI_Send(mssg, Nsize, MPI_INT, rank + 1, repeat, comm);
      MPI_Recv(mssg, Nsize, MPI_INT, numprocs - 1 , repeat, comm, &status);

      // MPI_Send(mssg, Nsize, MPI_INT, (rank+1) % numprocs, repeat, comm);
      // MPI_Recv(mssg, Nsize, MPI_INT, (rank+numprocs-1) % numprocs , repeat, comm, &status);

    } else {
      MPI_Recv(mssg, Nsize, MPI_INT, rank - 1, repeat, comm, &status);
      for (long idx = 0; idx < Nsize; ++idx) mssg[idx] += rank;
      MPI_Send(mssg, Nsize, MPI_INT, (rank+1) % numprocs, repeat, comm);

      // MPI_Recv(mssg, Nsize, MPI_INT, (rank+numprocs-1) % numprocs, repeat, comm, &status);
      // for (long idx = 0; idx < Nsize; ++idx) mssg[idx] += rank;
      // MPI_Send(mssg, Nsize, MPI_INT, (rank+1) % numprocs, repeat, comm);

    }
  }

  // printf("message_in is %d\n", *mssg);
  tt = MPI_Wtime() - tt;

  if(!rank) { printf("Final mssg value: %d | ", mssg[0]); }

  free(mssg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 2) {
    printf("Usage: mpirun -np 3 ./int-ring <N: num-of-loops> \n");
    abort();
  }
  
  long N = atoi(argv[1]), NSize = 1;
  int rank, numprocs, mssg = 0;

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);

  // printf("Loop number is is %d\n", N);
  // printf("Number of processes is %d\n", numprocs);

  double tt = execute_ring(NSize, N, comm);
  if (!rank) { printf("int-ring latency: %e ms\n", tt/N * 1000); }

  mssg = 0;
  tt = execute_ring(NSize, N, comm);
  if (!rank) { printf("int-ring bandwidth: %e GB/s\n", (sizeof(int)*NSize*N)/tt/1e9); }

  // 2MByte Array Communication
  long number_of_bytes = 2;
  long array_size = number_of_bytes * (1 << 20);
  NSize = array_size;

  tt = execute_ring(NSize, N, comm);
  if (!rank) { printf("large array latency: %e ms\n", tt/N * 1000); }
  
  tt = execute_ring(NSize, N, comm);
  if (!rank) { printf("large array bandwidth: %e GB/s\n", (sizeof(int)*NSize*N)/tt/1e9); }

  MPI_Finalize();

  return 0;
}

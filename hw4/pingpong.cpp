#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

// To run this program, use the following
// mpirun -np 2 ./pingpong 2 10

double time_pingpong(int proc0, int proc1, long Nrepeat, long Nsize, MPI_Comm comm) {

  int rank, numprocs, namelen;
  char* msg = (char*) malloc(Nsize);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);
  MPI_Get_processor_name(processor_name, &namelen);
  
  for (long i = 0; i < Nsize; i++) msg[i] = 42;

  printf("Rank %d on %s\n", rank, processor_name);

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (repeat % 2 == 0) { // even iterations
      if (rank == proc0) {
        MPI_Send(msg, Nsize, MPI_CHAR, proc1, repeat, comm);
        //printf("Rank %d on %s sent data to rank %d on %s with tag %d\n", rank, processor_name, proc0, processor_name, repeat);
      }
      else if (rank == proc1) {
        MPI_Recv(msg, Nsize, MPI_CHAR, proc0, repeat, comm, &status);
        //printf("Rank %d on %s received data to rank %d on %s with tag %d\n", rank, processor_name, proc1, processor_name, repeat);
      } 
    }
    else { // odd iterations
      if (rank == proc0) {
        MPI_Recv(msg, Nsize, MPI_CHAR, proc1, repeat, comm, &status);
        //printf("Rank %d on %s received data to rank %d on %s with tag %d\n", rank, processor_name, proc1, processor_name, repeat);
      }
      else if (rank == proc1) {
        MPI_Send(msg, Nsize, MPI_CHAR, proc0, repeat, comm);
        //printf("Rank %d on %s sent data to rank %d on %s with tag %d\n", rank, processor_name, proc0, processor_name, repeat);
      }
    }
  }

  tt = MPI_Wtime() - tt;

  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Usage: mpirun ./pingpong <process-rank0> <process-rank1>\n");
    abort();
  }
  int proc0 = atoi(argv[1]);
  int proc1 = atoi(argv[2]);

  int rank;
  
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  //MPI_Comm_size(comm, &numprocs);
  //MPI_Get_processor_name(processor_name, &namelen);
 
  long Nrepeat = 1000;
  double tt = time_pingpong(proc0, proc1, Nrepeat, 1, comm);
  if (!rank) printf("pingpong latency: %e ms\n", tt/Nrepeat * 1000);

  Nrepeat = 10000;
  long Nsize = 1000000;
  tt = time_pingpong(proc0, proc1, Nrepeat, Nsize, comm);
  if (!rank) printf("pingpong bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
}


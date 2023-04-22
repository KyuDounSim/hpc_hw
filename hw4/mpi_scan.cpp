#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

// from hw3 init code
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_mpi(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  int rank, numprocs, namelen;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);
  
}

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

int main() {
long n_array[4] = {100000000, 200000000, 500000000, 1000000000};
  long* A, *B0, *B1;

  double tt; long err;
  for(int i = 0 ; i < 4; ++i) {
    printf("N = %ld\n", n_array[i]);
    A = (long*) malloc(n_array[i] * sizeof(long));
    B0 = (long*) malloc(n_array[i] * sizeof(long));
    B1 = (long*) malloc(n_array[i] * sizeof(long));
    for (long i = 0; i < n_array[i]; i++) A[i] = rand();
    
    tt = omp_get_wtime();
    scan_seq(B0, A, n_array[i]);
    printf("sequential-scan = %fs\n", omp_get_wtime() - tt);
    
    err = 0;

    omp_set_num_threads(3);
    tt = omp_get_wtime();
    scan_omp(B1, A, n_array[i]);
    printf("parallel-scan(3thrd)   = %fs\n", omp_get_wtime() - tt);
    for (long i = 0; i < n_array[i]; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);

    omp_set_num_threads(8);
    tt = omp_get_wtime();
    scan_omp(B1, A, n_array[i]);
    printf("parallel-scan(8thrd)   = %fs\n", omp_get_wtime() - tt);
    for (long i = 0; i < n_array[i]; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);

    omp_set_num_threads(10);
    tt = omp_get_wtime();
    scan_omp(B1, A, n_array[i]);
    printf("parallel-scan(10thrd)   = %fs\n", omp_get_wtime() - tt);
    for (long i = 0; i < n_array[i]; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);
    
    omp_set_num_threads(20);
    tt = omp_get_wtime();
    scan_omp(B1, A, n_array[i]);
    printf("parallel-scan(20thrd)   = %fs\n", omp_get_wtime() - tt);
    for (long i = 0; i < n_array[i]; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);

    omp_set_num_threads(50);
    tt = omp_get_wtime();
    scan_omp(B1, A, n_array[i]);
    printf("parallel-scan(50thrd)   = %fs\n", omp_get_wtime() - tt);
    for (long i = 0; i < n_array[i]; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    printf("error = %ld\n", err);

    // reference printing A, B0, B1
    // for (long i = 0; i < N; ++i) printf("A[%ld] = %ld, B0[%ld] = %ld, B1[%ld] = %ld\n", i, A[i], i, B0[i], i, B1[i]);

    // free(A);
    // free(B0);
    // free(B1);
  }

  free(A);
  free(B0);
  free(B1);
  return 0;
}

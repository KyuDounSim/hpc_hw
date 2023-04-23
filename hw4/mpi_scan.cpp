#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <mpi.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
double scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return 0.0;
  double tt = MPI_Wtime();
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }

  tt = MPI_Wtime() - tt;
  return tt;
}

double scan_mpi(long* prefix_sum, const long* A, long n, MPI_Comm comm) {
  if (n == 0) return 0.0;

  int rank, numprocs;
  
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);
  
  MPI_Barrier(comm);

  double tt = MPI_Wtime();
  // assume that n % numprocs == 0
  int sub_array_size = n / numprocs;
  long* sub_array = (long *)malloc(sizeof(long) * sub_array_size);

  MPI_Scatter(A, sub_array_size, MPI_LONG, sub_array, sub_array_size, MPI_LONG, 0, comm);

  //for(int i = 0; i < sub_array_size; ++i) printf("Rank(%d) sub_array[%d]: %d\n", rank, i, sub_array[i]);

  long* sub_array_prefix_output = (long *)malloc(sizeof(long) * (sub_array_size + 1) );
  // compute local prefix sum of the sub array
  scan_seq(sub_array_prefix_output, sub_array, sub_array_size + 1);
  
  //for(int i = 0; i < sub_array_size + 1; ++i) printf("Rank(%d) sub_array_prefix_output[%d]: %d\n", rank, i, sub_array_prefix_output[i]);

  long* offset_array = (long*)malloc(sizeof(long) * ( numprocs ));
  MPI_Allgather(&sub_array_prefix_output[sub_array_size], 1, MPI_LONG, offset_array, 1, MPI_LONG, comm);

  // accumulate offsets
  long* offset_array_output = (long*)malloc(sizeof(long) * ( numprocs ));
  scan_seq(offset_array_output, offset_array, numprocs );
  
  //for(int i = 0; i < numprocs ; ++i) printf("Rank(%d) offset_array_output[%d]: %d\n", rank, i, offset_array_output[i]);

  for(long ii = 0; ii < sub_array_size; ++ii) {
    //printf("sub_array_prefix: %d + offset: %d\n", sub_array_prefix_output[ii], offset_array_output[rank]);
    sub_array_prefix_output[ii] += offset_array_output[rank];
  }
  
  //for(int i = 0; i < sub_array_size; ++i) printf("Rank(%d) sub_array_prefix_output[%d]: %d\n", rank, i, sub_array_prefix_output[i]);
  
  MPI_Gather(sub_array_prefix_output, sub_array_size, MPI_LONG, &(prefix_sum[0]), sub_array_size, MPI_LONG, 0, comm);

  tt = MPI_Wtime() - tt;

  free(sub_array);
  free(sub_array_prefix_output);
  free(offset_array);
  free(offset_array_output);
  
  return tt;
}

// Assume that the size of the array is divisible by
// the number of processes
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc != 2) {
    printf("Usage: mpirun ./mpi_scan <prefix-array-size> \n");
    abort();
  }
  
  long array_size = atoi(argv[1]);
  int rank, numprocs;

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &numprocs);
  
  long* A, *B0, *B1;
  double tt; long err;

  if(!rank) {
    printf("N = %ld\n", array_size);
    A = (long*) malloc(array_size * sizeof(long));
    B0 = (long*) malloc(array_size * sizeof(long));
    B1 = (long*) malloc(array_size * sizeof(long));
    //for (long ii = 0; ii < array_size; ++ii) A[ii] = ii;
    for (long ii = 0; ii < array_size; ++ii) A[ii] = rand();
  
    //for (long ii = 0; ii < array_size; ++ii) printf("A[%d]: %d\n", ii, A[ii]);
  
    tt = scan_seq(B0, A, array_size);
    printf("sequential-scan = %fs\n", tt);
    //for (long ii = 0; ii < array_size; ++ii) printf("B0[%d]: %d\n", ii, B0[ii]);
    err = 0;
  } 

  tt = scan_mpi(B1, A, array_size, comm);

  if(!rank) {
    printf("mpi-parallel-scan (%d processes) = %fs\n", numprocs, tt);
    for (long i = 0; i < array_size; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
    //for (long i = 0; i < array_size; i++) printf("B1[%d] == %d\n", i, B1[i]); 
    printf("error = %ld\n", err);

    free(A);
    free(B0);
    free(B1);
  }

  MPI_Finalize();
  return 0;
}


#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
// g++ -fopenmp omp-scan.cpp && ./a.out

// for my personal Mac
// g++-12 -fopenmp omp-scan.cpp && ./a.out

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  prefix_sum[0] = 0;
  #pragma omp parallel
  {
    int t = omp_get_thread_num(), p = omp_get_num_threads();
    
    // ranges
    long CHUNCK_SIZE = (n + p - 1) / p;
    long START_IDX = t * CHUNCK_SIZE;
    // last thread may have more elements if n % p != 0
    long END_IDX = (t == p - 1) ? n :(CHUNCK_SIZE * (t + 1));

    // sub array prefix computation
    long sum = 0;
    if (START_IDX != 0) {
      sum = A[START_IDX - 1];
    }

    for(long subArrayIdx = START_IDX + 1; subArrayIdx < END_IDX; ++subArrayIdx)
      sum += A[subArrayIdx - 1];
    
    prefix_sum[START_IDX] = sum;

    #pragma omp barrier

    // prefix sum computation, picking the beginning of each thread
    sum = 0;
    for(long thread_id = 0; thread_id < t; ++thread_id)
      sum += prefix_sum[thread_id * CHUNCK_SIZE];

    #pragma omp barrier

    prefix_sum[START_IDX] = sum + (START_IDX == 0 ? 0 : A[START_IDX - 1]);
    for(long p_sum_idx  = START_IDX + 1; p_sum_idx  < END_IDX; ++p_sum_idx)
      prefix_sum[p_sum_idx ] = prefix_sum[p_sum_idx - 1] + A[p_sum_idx - 1];

  }
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

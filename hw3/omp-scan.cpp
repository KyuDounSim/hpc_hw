#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
// g++ -fopenmp omp-scan.cpp && ./a.out

// on my personal Mac
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

// void scan_omp(long* prefix_sum, const long* A, long n) {


//   omp_set_num_threads(8);
//   printf("scan %d\n", omp_get_num_threads());

  // if (n == 0) return;

  // #pragma omp parallel
  // {
  //   int p = omp_get_num_threads();
  //   int t = omp_get_thread_num();
  //   printf("scan_omp: thread %d of %d \n", t, p);
  //   long CHUNCK_SIZE {0}, START_IDX {0}, END_IDX {0};
  //   long* offset = (long*) malloc((p) * sizeof(long));

  //   // parallel p chunks, each chunk has CHUNCK_SIZE elements
  //   // sequential scan on each chunk to compute the prefix sum  
  //   #pragma omp parallel private(t, CHUNCK_SIZE, END_IDX) shared(prefix_sum, A, n, offset, p) 
  //   {
  //     CHUNCK_SIZE = (n + p - 1) / p;
  //     START_IDX = (CHUNCK_SIZE * t) + 1;
  //     END_IDX = (t == p - 1) ? n - 1: CHUNCK_SIZE * (t + 1) ;

  //     // long CHUNCK_SIZE = n/p, i = t * CHUNCK_SIZE, 
  //     // j = (t == p-1)?(n-1):( t * CHUNCK_SIZE +CHUNCK_SIZE-1);

  //     // END_IDX = CHUNCK_SIZE * (t + 1);
  //     // END_IDX = (CHUNCK_SIZE * (t + 1) < n) ? CHUNCK_SIZE * (t + 1) : n - 1 ;
  //     // printf("CHUNCK_SIZE = %ld scan_omp: thread %d of %d\n", CHUNCK_SIZE, t, p);
  //     // printf("scan_omp: thread %d of %d has start idx %d, end_idx %d, \n", t, p
  //     // , START_IDX, END_IDX);

  //     prefix_sum[0] = 0;
  //     for (long i = START_IDX; i <= END_IDX; ++i)
  //     {
  //       prefix_sum[i] = A[i] + A[i - 1];
  //       printf("prefix_sum[%ld] = %ld \n", i, prefix_sum[i]);
  //     }
  //     #pragma omp barrier

  //     if (t == 0) {
  //       offset[t] = 0;
  //     } else {
  //       offset[t] = prefix_sum[END_IDX] + A[END_IDX];
  //     }
  //     #pragma omp barrier
    
  //   // #pragma omp single
  //   // for(int i = 0; i < p; ++i)
  //   // {
  //   //   printf("offset[%d] = %ld\n", i, offset[i]);
  //   // }
  //   // accumulate the offset
  //   #pragma omp single
  //   offset[0] = 0;
  //   for(int i = 1; i < p; ++i)
  //   {
  //     offset[i] += offset[i-1];
  //     printf("offset[%d] = %ld\n", i, offset[i]);
  //   }
  //   }
  // }
  // }

int main() {
  // long N = 100000000;
  long N = 100;
  // long N = 11;

  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  // for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) A[i] = i;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  long err = 0;

  omp_set_num_threads(4);
  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan(4thrd)   = %fs\n", omp_get_wtime() - tt);
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  // omp_set_num_threads(8);
  // tt = omp_get_wtime();
  // scan_omp(B1, A, N);
  // printf("parallel-scan(8thrd)   = %fs\n", omp_get_wtime() - tt);
  // for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  // printf("error = %ld\n", err);

  // omp_set_num_threads(10);
  // tt = omp_get_wtime();
  // scan_omp(B1, A, N);
  // printf("parallel-scan(10thrd)   = %fs\n", omp_get_wtime() - tt);
  // for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  // printf("error = %ld\n", err);
  
  // omp_set_num_threads(20);
  // tt = omp_get_wtime();
  // scan_omp(B1, A, N);
  // printf("parallel-scan(20thrd)   = %fs\n", omp_get_wtime() - tt);
  // for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  // printf("error = %ld\n", err);

  // omp_set_num_threads(50);
  // tt = omp_get_wtime();
  // scan_omp(B1, A, N);
  // printf("parallel-scan(50thrd)   = %fs\n", omp_get_wtime() - tt);
  // for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  // printf("error = %ld\n", err);

  // reference printing A, B0, B1
  // for (long i = 0; i < N; ++i) printf("A[%ld] = %ld, B0[%ld] = %ld, B1[%ld] = %ld\n", i, A[i], i, B0[i], i, B1[i]);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

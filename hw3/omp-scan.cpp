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

void scan_omp(long* prefix_sum, const long* A, long n) {
  // int p = omp_get_num_threads();
  // int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();
  long CHUNCK_SIZE {0}, END_IDX {0};
  long* offset;

  if (n == 0) return;
  // parallel p chunks, each chunk has CHUNCK_SIZE elements
  // sequential scan on each chunk to compute the prefix sum  
  #pragma omp parallel private(p, t, CHUNCK_SIZE, END_IDX) shared(prefix_sum, A, n, offset) 
  {
     t = omp_get_thread_num();
     p = omp_get_num_threads();
    offset = (long*) malloc((p + 1) * sizeof(long));
    printf("scan_omp: thread %d of %d %\n", t, p);
    // printf("scan_omp: total thred num: %d\n", p);
    CHUNCK_SIZE = n / p;
    END_IDX = (t == p - 1) ? n - 1: CHUNCK_SIZE * (t + 1) ;

    // long CHUNCK_SIZE = n/p, i = t * CHUNCK_SIZE, 
    // j = (t == p-1)?(n-1):( t * CHUNCK_SIZE +CHUNCK_SIZE-1);

    // END_IDX = CHUNCK_SIZE * (t + 1);
    // END_IDX = (CHUNCK_SIZE * (t + 1) < n) ? CHUNCK_SIZE * (t + 1) : n - 1 ;
    // printf("CHUNCK_SIZE = %ld scan_omp: thread %d of %d\n", CHUNCK_SIZE, t, p);
    prefix_sum[0] = 0;
    for (long i = CHUNCK_SIZE * t + 1; i <= END_IDX; ++i)
    {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      printf("prefix_sum[%ld] = %ld \n", i, prefix_sum[i]);
    }
    #pragma omp barrier
    if (t == 0) {
      offset[t] = 0;
    } else {
      offset[t] = prefix_sum[CHUNCK_SIZE * t - 1] + A[CHUNCK_SIZE * t - 1];
    }
    #pragma omp barrier
    
    // #pragma omp single
    // for(int i = 0; i < p; ++i)
    // {
    //   printf("offset[%d] = %ld\n", i, offset[i]);
    // }
    // accumulate the offset
    #pragma omp single
    offset[0] = 0;
    for(int i = 1; i < p; ++i)
    {
      offset[i] += offset[i-1];
      printf("offset[%d] = %ld\n", i, offset[i]);
    }
    // for (long i = CHUNCK_SIZE * t + 1; i <= END_IDX ; ++i)
    // {
    //   prefix_sum[i] = prefix_sum[i] + offset[t];
    //   printf("final prefix_sum[%ld] = %ld + %ld \n", i, prefix_sum[i], offset[t]);
    // }
  }
}

int main() {
  // long N = 100000000;
  // long N = 100;
  long N = 16;


  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  // for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) A[i] = i;
  // for (long i = 0; i < N; i++) B0[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  long err = 0;

  omp_set_num_threads(3);
  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan(4thrd)   = %fs\n", omp_get_wtime() - tt);
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  // omp_set_num_threads(10);
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

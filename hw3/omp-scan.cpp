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
  int p = omp_get_num_threads();
  int t = omp_get_thread_num();
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel

  if (n == 0) return;
  prefix_sum[0] = 0;

  long CHUNCK_SIZE {0}, END_IDX {0};

  // parallel p chunks, each chunk has CHUNCK_SIZE elements
  // sequential scan on each chunk to compute the prefix sum  
  #pragma omp parallel private(p, t, CHUNCK_SIZE) shared(prefix_sum, A, n) 
  {
    t = omp_get_thread_num();
    p = omp_get_num_threads();
    CHUNCK_SIZE = (n + p - 1) / p;
    END_IDX = CHUNCK_SIZE * (t + 1);

    printf("CHUNCK_SIZE = %ld scan_omp: thread %d of %d\n", CHUNCK_SIZE, t, p);
    
    if(t == 0) {
      printf("Under thread %d of %d indexes are \n", t, p);

      for (long i = CHUNCK_SIZE * t + 1; i < END_IDX; i++)
      {
        printf("%ld ", i);
      }
      printf("\n");

      // prefix_sum[i] = prefix_sum[i-1] + A[i-1];
      // prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }

  // // wait until all threads finish the sequential scan
  // #pragma barrier


  // just to make the error 0
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

int main() {

  #pragma omp parallel num_threads(4)
  {
    printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
  printf("\n");
  #pragma omp parallel num_threads(8)
  {
    printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
  printf("\n");

  #pragma omp parallel num_threads(12)
  {
    printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }
  printf("\n");

  // long N = 100000000;
  long N = 100;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

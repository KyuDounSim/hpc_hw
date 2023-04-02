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
  long* offset = (long*) malloc(p * sizeof(long));

  // parallel p chunks, each chunk has CHUNCK_SIZE elements
  // sequential scan on each chunk to compute the prefix sum  
  #pragma omp parallel default(none) private(p, t, CHUNCK_SIZE, END_IDX) shared(prefix_sum, A, n, offset) 
  {
    t = omp_get_thread_num();
    p = omp_get_num_threads();
    CHUNCK_SIZE = (n + p - 1) / p;
    END_IDX = CHUNCK_SIZE * (t + 1);
    // END_IDX = (CHUNCK_SIZE * (t + 1) < n) ? CHUNCK_SIZE * (t + 1) : n - 1 ;
    // printf("CHUNCK_SIZE = %ld scan_omp: thread %d of %d\n", CHUNCK_SIZE, t, p);

    for (long i = CHUNCK_SIZE * t + 1; i < END_IDX ; ++i)
    {
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }
  }

  #pragma omp barrier

  // compute the offset for each chunk
  #pragma omp parallel default(none) private(p, t, CHUNCK_SIZE, END_IDX) shared(prefix_sum, A, n, offset) 
  {
    t = omp_get_thread_num();
    p = omp_get_num_threads();
    CHUNCK_SIZE = (n + p - 1) / p;
    END_IDX = CHUNCK_SIZE * (t + 1);
    // END_IDX = (CHUNCK_SIZE * (t + 1) < n) ? CHUNCK_SIZE * (t + 1) : n - 1 ;
    // printf("CHUNCK_SIZE = %ld scan_omp: thread %d of %d\n", CHUNCK_SIZE, t, p);

    if (t == 0) {
      offset[t] = 0;
    } else {
      offset[t] = prefix_sum[CHUNCK_SIZE * t - 1] + A[CHUNCK_SIZE * t - 1];
    }

    // accumulate the offset
    #pragma omp single
    for(int i = 1; i < p; ++i)
    {
      offset[i] += offset[i-1];
    }
    // printf("offset[%d] = %ld\n", t, offset[t]);
  }

  #pragma omp parallel default(none) private(p, t, CHUNCK_SIZE, END_IDX) shared(prefix_sum, A, n, offset) 
  {
    t = omp_get_thread_num();
    p = omp_get_num_threads();
    CHUNCK_SIZE = (n + p - 1) / p;
    END_IDX = CHUNCK_SIZE * (t + 1);

    for (long i = CHUNCK_SIZE * t + 1; i < END_IDX ; ++i)
    {
      // printf("prefix_sum[%ld] = %ld + %ld \n", i, prefix_sum[i], offset[t]);
      prefix_sum[i] += offset[t];
    }
  }

  // printf("offset calculation: %ld + %ld \n", prefix_sum[END_IDX - 1], A[END_IDX - 1]);
  // offset[t] = prefix_sum[END_IDX - 1] + A[END_IDX - 1];
  // printf("offset[%d] = %ld\n", t, offset[t]);

  // reference answer
  // prefix_sum[0] = 0;
  // for (long i = 1; i < n; i++) {
  //   prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  // }
  free(offset);
}

int main() {

  // #pragma omp parallel num_threads(4)
  // {
  //   printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  // }
  // printf("\n");
  // #pragma omp parallel num_threads(8)
  // {
  //   printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  // }
  // printf("\n");

  // #pragma omp parallel num_threads(12)
  // {
  //   printf("omp-scan from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  // }
  // printf("\n");

  long N = 100000000;
  // long N = 100;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  // for (long i = 0; i < N; i++) A[i] = rand();
  // for simpler testing
  for (long i = 0; i < N; i++) A[i] = i;
  for (long i = 0; i < N; i++) B1[i] = 0;
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  // reference printing A, B0, B1
  // for (long i = 0; i < N; ++i) printf("A[%ld] = %ld, B0[%ld] = %ld, B1[%ld] = %ld\n", i, A[i], i, B0[i], i, B1[i]);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

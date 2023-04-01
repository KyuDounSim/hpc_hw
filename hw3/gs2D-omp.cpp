// gs2D-omp
// works with or without OPENMP
// $ g++ -fopenmp gs2D-omp.cpp && ./a.out
// $ g++ gs2D-omp.cpp $$ ./a.out

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0;}
inline omp_int_t omp_get_num_threads() { return 1;}
#endif

#include <algorithm>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
  printf("maximum number of threads = %d\n", omp_get_num_threads());

  // pragma is ignored if command is undefined
  #pragma omp parallel
  {
    printf("hello world from thread %d of %d\n", omp_get_thread_num(), omp_get_num_threads());
  }

  return 0;
}

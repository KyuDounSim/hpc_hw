// $ g++ -std=c++11 -O3 -march=native compute.cpp && ./a.out -n 1000000000
// $ g++ -std=c++11 -O3 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized && ./a.out -n 1000000000

// compile with name
// $ g++ -std=c++11 -O3 -march=native compute.cpp -o compute_vanilla && ./compute_vanilla -n 1000000000
// $ g++ -std=c++11 -O3 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized -o compute_vector && ./compute_vector -n 1000000000

// compile and output with O0 flag
// $ g++ -std=c++11 -O0 -march=native compute.cpp -o compute_vanilla && ./compute_vanilla -n 1000000000 > outputs/compute_vanilla_output_O0
// $ g++ -std=c++11 -O0 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized -o compute_vector && ./compute_vector -n 1000000000 > outputs/compute_vector_output_O0

// compile and output with O3 flag
// $ g++ -std=c++11 -O3 -march=native compute.cpp -o compute_vanilla && ./compute_vanilla -n 1000000000 > outputs/compute_vanilla_output_O3
// $ g++ -std=c++11 -O3 -march=native compute.cpp -ftree-vectorize -fopt-info-vec-optimized -o compute_vector && ./compute_vector -n 1000000000 > outputs/ompute_vector_output_O3

#include <stdio.h>
#include <math.h>
#include "utils.h"

#define CLOCK_FREQ 2.8e9

void compute_fn_mult_add(double* A, double B, double C) {
  (*A) = (*A) * B + C;
}

void compute_fn_div(double* A, double B, double C) {
  (*A) = C / (*A);
}

void compute_fn_sqrt(double* A, double B, double C) {
  (*A) = sqrt(*A);
}

void compute_fn_sin(double* A, double B, double C) {
  (*A) = sin(*A);
}

int main(int argc, char** argv) {
  Timer t;
  long repeat = read_option<long>("-n", argc, argv);
  double A = 1.5;
  double B = 1./2;
  double C = 2.;

  printf("\ncompute_fn_mult_add computed.\n");
  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn_mult_add(&A, B, C);
  printf("%f seconds\n", t.toc());
  printf("%f cycles/eval\n", t.toc()*CLOCK_FREQ/repeat);
  printf("%f Gflop/s\n\n", 2*repeat/1e9/t.toc());

  printf("compute_fn_div computed.\n");
  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn_div(&A, B, C);
  printf("%f seconds\n", t.toc());
  printf("%f cycles/eval\n", t.toc()*CLOCK_FREQ/repeat);
  printf("%f Gflop/s\n\n", repeat/1e9/t.toc());

  printf("compute_fn_sqrt computed.\n");
  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn_sqrt(&A, B, C);
  printf("%f seconds\n", t.toc());
  printf("%f cycles/eval\n", t.toc()*CLOCK_FREQ/repeat);
  printf("%f Gflop/s\n\n", repeat/1e9/t.toc());
  
  printf("compute_fn_sin computed.\n");
  t.tic();
  for (long i = 0; i < repeat; i++) compute_fn_sin(&A, B, C);
  printf("%f seconds\n", t.toc());
  printf("%f cycles/eval\n", t.toc()*CLOCK_FREQ/repeat);
  printf("%f Gflop/s\n\n", repeat/1e9/t.toc());
  
  return A;
}

// Synopsis
//
// By design, this computation is such that only one fused-multiply-accumulate
// instruction can execute at one time i.e. this computation cannot be
// vectorized or pipelined. Therefor, this example can be used to measure
// latency of operations.
//
// * Compare the observed latency with the expected latency for _mm256_fmadd_pd
// instruction for your architecture from this link:
// (https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_fmadd_pd&expand=2508)
//
// * Try replacing the mult-add operation with some other computation, like
// division, sqrt, sin, cos etc. to measure the latency of those operations and
// compare it with the latency of mult-add.

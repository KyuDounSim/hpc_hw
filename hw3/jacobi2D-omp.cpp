// jacobi2D-omp
// works with or without OPENMP
// $ g++ -fopenmp jacobi2D-omp.cpp && ./a.out
// $ g++ jacobi2D-omp.cpp $$ ./a.out

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

#include "utils.h"

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *u, double* f, int N, double invhsq, double a, double b)
{
  int i, j;
  double tmp, res = 0.0;
  double left = u[j - 1];    
  double right = u[j + 1];
  double up = u[j - N];   
  double down = u[j + N];

  #pragma omp parallel for reduction (+:res)
  for (i = 1; i <= N; ++i) {
    for (j = 1; j <= N; ++j) {
      // tmp = ((2.0*u[i] - u[i-1] - u[i+1]) * invhsq - 1);
      tmp = a * u[i * N + j] + b * u[i * N + j - 1] + b * u[i * N + j + 1] + b * u[(i - 1) * N + j] + b * u[(i + 1) * N + j];
      res += (tmp * tmp);
    }
  }
  return sqrt(res);
}	

int main(int argc, char** argv) {
  printf("maximum number of threads = %d\n", omp_get_max_threads());
  int i, N, iter, max_iters;

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* timing */
  double t = omp_get_wtime();

  /* Allocation of vectors, including left and right ghost points */
  // double* u     = (double *) calloc(sizeof(double), (N + 2) * (N + 2));
  // double* u_new = (double *) calloc(sizeof(double), (N + 2) * (N + 2));
  // double* f     = (double *) calloc(sizeof(double), N * N);

  double* u     = new double    [(N + 2) * (N + 2)];
  double* u_new = new double    [(N + 2) * (N + 2)];
  double* f     = new double    [ N      *  N     ];

  for(int i = 0; i < N * N; ++i){ f[i] = 1.0; }

  double h = 1.0 / (N + 1.0);
  double hsq = h*h;
  double invhsq = 1./hsq;
  double a = 4./hsq, b = -1./hsq;

  double res, res0, tol = 1e-5;

  /* initial residual */
  res0 = compute_residual(u, f, N, invhsq, a, b);
  res = res0;
  u[0] = u[N+1] = 0.0;
  double omega = 1.0; //2./3;

  for (iter = 0; iter < max_iters && res/res0 > tol; iter++) {

    /* Jacobi step for all the inner points */
    #pragma omp parallel for
    for (i = 1; i <= N; i++){
      u_new[i] =  u[i] + omega * 0.5 * (hsq + u[i - 1] + u[i + 1] - 2*u[i]);
    }

    /* flip pointers; that's faster than memcpy  */
    // memcpy(u,unew,(N+2)*sizeof(double));
    double* utemp = u;
    u = u_new;
    u_new = utemp;
    if (0 == (iter % 1)) {
      res = compute_residual(u, N, invhsq);
      printf("Iter %d: Residual: %g\n", iter, res);
    }
  }

  /* Clean up */
  // free(u);
  // free(u_new);
  delete[] u;
  delete[] u_new;

  /* timing */
  t = omp_get_wtime() - t;
  printf("Time elapsed is %f.\n", t);

  return 0;
}

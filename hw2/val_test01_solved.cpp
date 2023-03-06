# include <cstdlib>
# include <iostream>

using namespace std;

int main ( );
void f ( int n );

int main ( )

{
  int n = 10;

  cout << "\n";
  cout << "TEST01\n";
  cout << "  C++ version.\n";
  cout << "  A sample code for analysis by VALGRIND.\n";

  f ( n );

  cout << "\n";
  cout << "TEST01\n";
  cout << "  Normal end of execution.\n";

  return 0;
}

void f ( int n )

{
  int i;
  int *x;

  x = ( int * ) malloc ( n * sizeof ( int ) );

  x[0] = 1;
  cout << "  " << 0 << "  " << x[0] << "\n";

  x[1] = 1;
  cout << "  " << 1 << "  " << x[1] << "\n";

  // For loop condition used  to be "i <= n", but this creates an out-of-bound
  // memory access of the array x, should be changed to "i < n".
  for ( i = 2; i < n; i++ )
  {
    x[i] = x[i-1] + x[i-2];
    cout << "  " << i << "  " << x[i] << "\n";
  }

  // free(x) is used instead of delete[] x. free() should be used for
  // memory allocated using malloc
  free(x);

  return;
}

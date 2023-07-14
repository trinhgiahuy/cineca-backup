#include <stdio.h>

int main(int argc, char const *argv[]){
  double err;

  #pragma omp target map(tofrom:Anew) map(tofrom:A)
  while (err >  tol && itter < itter_max){
    err = 0.0; 


  }
}

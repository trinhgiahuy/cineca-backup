#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Grid boundary conditions
#define RIGHT 1.0
#define LEFT 1.0
#define TOP 1.0
#define BOTTOM 10.0

// Algorithm settings
#define NPRINT 1000
#define MAX_ITER 100000
#define TOLERANCE 0.0001

void grid_init(double *grid,double *grid_new,int nx, int ny); 


int main(int argc, char*argv[]) {

  int i,j,k;
  double tmpnorm,bnorm,norm;

  if (argc !=3) {
      printf("usage: $argv[0] GRIDX GRIDY \n");
      return(1);
  }

  int nx=atoi(argv[1]);
  int ny=atoi(argv[2]);
  int ny2=ny+2;
  int nelems=(nx+2)*(ny+2);

  printf("grid size %d X %d \n",ny,ny);

// Allocate memory for current and new grids
  double *temp;
  double *grid= (double*)malloc(sizeof(double)*nelems);
  double *grid_new= (double*)malloc(sizeof(double)*nelems);

  // initialise grids
  grid_init(grid,grid_new,nx,ny); 

  // calculate norm factor
  tmpnorm=0.0;
  for (i=1;i<=nx;i++) {
    for (j=1;j<=ny;j++) {
      k=(ny+2)*i+j;            
      tmpnorm=tmpnorm+(double)pow(grid[k]*4.0-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2); 
    }
  }
  bnorm=sqrt(tmpnorm);

//    MAIN LOOP 
  int iter;
  for (iter=0; iter<MAX_ITER; iter++) {

    tmpnorm=0.0;

    for (i=1;i<=nx;i++) {
     for (j=1;j<=ny;j++) {
      k=(ny+2)*i+j;
      tmpnorm=tmpnorm+pow(grid[k]*4.0-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2); 
    }
  }
   
    // calculate norm factor
    norm=sqrt(tmpnorm)/bnorm;

    if (norm < TOLERANCE) break;

    for (i=1;i<=nx;i++) {
      for (j=1;j<=ny;j++) {
        k=(ny+2)*i+j;    
	grid_new[k]=0.25 * (grid[k-1]+grid[k+1] + grid[k-(ny+2)] + grid[k+(ny+2)]);
      }
    }

      // swap current and new grid pointers
      temp=grid;
      grid=grid_new;
      grid_new=temp;

    if (iter % NPRINT ==0) printf("Iteration =%d ,Relative norm=%e\n",iter,norm);
  }

  printf("Terminated on %d iterations, Relative Norm=%e \n", iter,norm);
  
  // free memory resources
  free(grid);
  free(grid_new);


  return 0;
    

  }

// Initialise Grids 
void grid_init(double *grid,double *grid_new,int nx, int ny) {

  int i,j,k;

  // top and bottom boundaries
  for (i=0;i<ny+2;i++) {
    grid_new[i]=grid[i]=TOP;
    j=(ny+2)*(nx+1)+i;
    grid_new[j]=grid[j]=BOTTOM;
  }

  // left and right boundaries
  for (i=1;i<nx+1;i++) {
    j=(ny+2)*i;
    grid_new[j]=grid[j]=LEFT;
    grid_new[j+ny+1]=grid[j+ny+1]=RIGHT;
  }

  // Initialise rest of grid
  for (i=1;i<=nx;i++)
    for (j=1;j<=ny;j++) {
      k=(ny+2)*i+j;
      grid_new[k]=grid[k]=0.0;
    }

} // end grid_init



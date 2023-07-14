#include <stdio.h>
#include <math.h>

// Grid boundary conditions
#define RIGHT 1.0
#define LEFT 1.0
#define TOP 1.0
#define BOTTOM 10.0

// Algorithm settings
#define TOLERANCE 0.0001
#define NPRINT 1000
#define MAX_ITER 1000000

int main(int argc, char*argv[]) {

  int k;
  double tmpnorm,bnorm,norm;

  if (argc !=3) {
    printf("usage: $argv[0] GRIDX GRIDY\n");
      return(1);
  }

  int nx=atoi(argv[1]);
  int ny=atoi(argv[2]);
  int ny2=ny+2;

  printf("grid size %d X %d \n",ny,ny);
  double *grid= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));
  double *grid_new= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));
  double *temp= (double*)malloc(sizeof(double)*(nx+2)*(ny+2));

  // Initialise Grid boundaries
  int i,j;
  for (i=0;i<ny+2;i++) {
    grid_new[i]=grid[i]=TOP;
    j=(ny+2)*(nx+1)+i;
    grid_new[j]=grid[j]=BOTTOM;
  }
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
/*  for (i=0;i<=nx+1;i++) {
    for (j=0;j<=ny+1;j++){
      printf("->%lf ",grid[j+i*(ny+2)]);
    }
    printf("\n");
  }
*/

  tmpnorm=0.0;
  for (i=1;i<=nx;i++) {
    for (j=1;j<=ny;j++) {
      k=(ny+2)*i+j;            
      tmpnorm=tmpnorm+pow(grid[k]*4-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2); 

    }
  }
  bnorm=sqrt(tmpnorm);

  int iter;
  for (iter=0; iter<MAX_ITER; iter++) {

    tmpnorm=0.0;
    for (i=1;i<=nx;i++) {
     for (j=1;j<=ny;j++) {
      k=(ny+2)*i+j;
      tmpnorm=tmpnorm+pow(grid[k]*4-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2); 
    }
  }
    
    norm=sqrt(tmpnorm)/bnorm;
    //printf("tmpnorm=%lf norm=%lf\n",tmpnorm,norm);
    if (norm < TOLERANCE) break;

    for (i=1;i<=nx;i++) {
      for (j=1;j<=ny;j++) {
        k=(ny+2)*i+j;    
	grid_new[k]=0.25 * (grid[k-1]+grid[k+1] + grid[k-(ny+2)] + grid[k+(ny+2)]);
      }
    }
    memcpy(temp, grid_new, sizeof(double) * (nx + 2) * (ny+2));
    memcpy(grid_new, grid, sizeof(double) * (nx + 2) * (ny+2));
    memcpy(grid, temp, sizeof(double) * (nx + 2) * (ny+2));

    if (iter % NPRINT ==0) printf("Iteration =%d ,Relative norm=%e\n",iter,norm);
  }

  printf("Terminated on %d iterations, Relative Norm=%e \n", iter,norm);
  
//  for (i=0;i<=nx+1;i++) {
//    for (j=0;j<=ny+1;j++){
//     printf("->%lf ",grid[j+i*(ny+2)]);
//    }
//    printf("\n");
//  }

  free(grid);
  free(temp);
  free(grid_new);
  return 0;
    

  }

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

// CUDA kernel
__global__
void stencil_sum(double*grid, double *grid_new, int nx, int ny)
{
  int index=blockIdx.x * blockDim.x +threadIdx.x; // global thread id

  int nrow=index/ny;
  int diff=index-(nrow*ny);
  int k=(nrow+1)*(ny+2)+diff+1;

  if (index<nx*ny)
      grid_new[k]=0.25 * (grid[k-1]+grid[k+1] + grid[k-(ny+2)] + grid[k+(ny+2)]);
}

// kernel for performing the normalisation

__global__
void stencil_norm(double*grid, double*arraynorm, int nx, int ny)
{
  int index=blockIdx.x * blockDim.x +threadIdx.x; // globEl thread id

  int nrow=index/ny;
  int diff=index-(nrow*ny);
  int k=(nrow+1)*(ny+2)+diff+1;

  if (index<nx*ny)
     arraynorm[index]=(double)pow(grid[k]*4.0-grid[k-1]-grid[k+1] - grid[k-(ny+2)] - grid[k+(ny+2)], 2);

}


int main(int argc, char*argv[]) {

  int i,j,k;
  double tmpnorm,bnorm,norm;

  if (argc !=3) {
      printf("usage: $argv[0] GRIDX GRIDY \n");
      return(1);
  }

  int nx=atoi(argv[1]);
  int ny=atoi(argv[2]);
  int nelems=(nx+2)*(ny+2);

  printf("grid size %d X %d \n",ny,ny);

// Allocate memory for current and new grids
  double *grid= (double*)malloc(sizeof(double)*nelems);
  double *grid_new= (double*)malloc(sizeof(double)*nelems);
  double *arraynorm= (double*)malloc(sizeof(double)*nx*ny);

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

//  CUDA
//  Allocate device memory.
  double *d_grid, *d_grid_new,*d_arraynorm;
  cudaMalloc(&d_grid,nelems*sizeof(double));
  cudaMalloc(&d_grid_new,nelems*sizeof(double));
  cudaMalloc(&d_arraynorm,nx*ny*sizeof(double));

  cudaMemcpy(d_grid_new,grid_new,nelems*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_grid,grid,nelems*sizeof(double), cudaMemcpyHostToDevice);


  // set kernel parameters
  int blockSize=256;
  int numBlocks = ((nx*ny)+blockSize-1)/blockSize;
  printf("numBlocks=%d\n",numBlocks);


//    MAIN LOOP 
  int iter;
  for (iter=0; iter<MAX_ITER; iter++) {


   stencil_norm<<<numBlocks,blockSize>>>(d_grid,d_arraynorm,nx,ny);
   cudaMemcpy(arraynorm,d_arraynorm,nx*ny*sizeof(double),cudaMemcpyDeviceToHost);   
  
   tmpnorm=0.0;
   for (i=0;i<nx*ny;i++)
      tmpnorm=tmpnorm+arraynorm[i];
 
    // calculate norm factor
    norm=(double)sqrt(tmpnorm)/bnorm;

    if (norm < TOLERANCE) break;

    // CUDA
    // grid update sent to GPU

    stencil_sum<<<numBlocks,blockSize>>>(d_grid,d_grid_new,nx,ny);

  // Wait for GPU to finish
   cudaDeviceSynchronize();

    double *temp=d_grid_new;
    d_grid_new=d_grid;
    d_grid=temp;

    if (iter % NPRINT ==0) printf("Iteration =%d ,Relative norm=%e\n",iter,norm);
  }

  printf("Terminated on %d iterations, Relative Norm=%e \n", iter,norm);
  
  // free memory resources
  free(grid);
  free(grid_new);
  free(arraynorm);

  //CUDA
  // Free CUDA resources 
  cudaFree(d_grid);
  cudaFree(d_grid_new);
  cudaFree(d_arraynorm);


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



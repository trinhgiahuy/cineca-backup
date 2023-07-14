
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


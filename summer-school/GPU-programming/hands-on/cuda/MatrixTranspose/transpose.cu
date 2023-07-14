#include <stdio.h>

#define MATRIXSIZE 448

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    32

// Number of repetitions used for timing averages. 
#define NUM_REPS  100

// OUTER: repeat over kernel launches 
// INNER: repeat inside the kernel over just the loads and stores
#define INNER yes
//#define OUTER yes

#define CUDA_CHECK( call )               \
{                                       \
cudaError_t cuerror = call;              \
if ( cudaSuccess != cudaSuccess )            \
   printf ("CUDA ERROR in %s %s : %s\n", __FILE__, __LINE__, cudaGetErrorString( cuerror ));                        \
}


// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copy(float *odata, float* idata, int width, int height, int nreps)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  
  int index  = xIndex + width*yIndex;
  for (int r=0; r < nreps; r++) {
    odata[index] = idata[index];
  }
}


// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void transposeNaive(float *odata, float* idata, int width, int height, int nreps)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;
  for (int r=0; r < nreps; r++) {
    odata[index_out] = idata[index_in];
  }
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    tile[threadIdx.y][threadIdx.x] = idata[index_in];
  
    __syncthreads();
  
    odata[index_out] = tile[threadIdx.x][threadIdx.y];
  }
}

// Coalesced transpose with no bank conflicts

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height, int nreps)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;  
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int r=0; r < nreps; r++) {
    tile[threadIdx.y][threadIdx.x] = idata[index_in];
  
    __syncthreads();
  
    odata[index_out] = tile[threadIdx.x][threadIdx.y];
  }
}


// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float* gold, float* idata,
			  const  int size_x, const  int size_y)
{
  for(  int y = 0; y < size_y; ++y) {
    for(  int x = 0; x < size_x; ++x) {
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
    }
  }
}

bool compare_results(float* gold, float* odata, int size)
{
  float thresold = 1e-3;
  for(  int x = 0; x < size; ++x) {
    if (abs(gold[x]-odata[x]) > thresold) {
      return false;
    }
  }
  return true;
}

int main( int argc, char** argv) 
{
  int size_x = MATRIXSIZE;
  int size_y = MATRIXSIZE;

  if (size_x%TILE_DIM != 0 || size_y%TILE_DIM != 0) {
    printf("\nMatrix size must be integral multiple of tile size\nExiting...\n\n");
    printf("FAILED\n\n");
    return 1;
  }

  // kernel pointer and descriptor
  void (*kernel)(float *, float *, int, int, int);
  char *kernelName;

  // execution configuration parameters
  dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM), threads(TILE_DIM,TILE_DIM);

  // CUDA events
  cudaEvent_t start, stop;

  // size of memory required to store the matrix
  const  int mem_size = sizeof(float) * size_x*size_y;

  // allocate host memory
  float *h_idata = (float*) malloc(mem_size);
  float *h_odata = (float*) malloc(mem_size);
  float *transposeGold = (float *) malloc(mem_size);  
  float *gold;

  // allocate device memory
  float *d_idata, *d_odata;
  CUDA_CHECK( cudaMalloc( (void**) &d_idata, mem_size) );
  CUDA_CHECK( cudaMalloc( (void**) &d_odata, mem_size) );

  // initalize host data
  for(  int i = 0; i < (size_x*size_y); ++i)
    h_idata[i] = (float) i;
  
  // copy host data to device
  CUDA_CHECK( cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice) );

  // Compute reference transpose solution
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n", 
	 size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM);

  // initialize events
  CUDA_CHECK( cudaEventCreate(&start) );
  CUDA_CHECK( cudaEventCreate(&stop) );

  //
  // loop over different kernels
  //

  bool success = true;

  for (int k = 0; k<4; k++)
  {
    // set kernel pointer
    switch (k) {
    case 0:
      kernel = &copy;                           kernelName = "simple copy       "; break;
    case 1:
      kernel = &transposeNaive;                 kernelName = "naive             "; break;
    case 2:
      kernel = &transposeCoalesced;             kernelName = "coalesced         "; break;
    case 3:
      kernel = &transposeNoBankConflicts;       kernelName = "optimized         "; break;
    }      

    // set reference solution
    if (kernel == &copy) {
      gold = h_idata;
    } else {
      gold = transposeGold;
    }

    // warmup to avoid timing startup
    kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);
    bool res;
#ifdef OUTER
    // take measurements for loop over kernel launches
    CUDA_CHECK( cudaEventRecord(start, 0) );
    for (int i=0; i < NUM_REPS; i++) {
      kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, 1);
    }
    CUDA_CHECK( cudaEventRecord(stop, 0) );
    CUDA_CHECK( cudaEventSynchronize(stop) );
    float outerTime;
    CUDA_CHECK( cudaEventElapsedTime(&outerTime, start, stop) );    

    CUDA_CHECK( cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost) );
    res = compare_results(gold, h_odata, size_x*size_y);
    if (res == false) {
      printf("*** %s kernel FAILED ***\n", kernelName);
      success = false;
    }

    // report effective bandwidths
    float outerBandwidth = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(outerTime/NUM_REPS);

    printf("transpose-Outer-%s, Throughput = %9.4f GB/s, Time = %.5f s, Size = %u fp32 elements\n", 
           kernelName, outerBandwidth, outerTime/NUM_REPS, (size_x * size_y));
#endif

#ifdef INNER
    // take measurements for loop inside kernel
    CUDA_CHECK( cudaEventRecord(start, 0) );
    kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y, NUM_REPS);
    CUDA_CHECK( cudaEventRecord(stop, 0) );
    CUDA_CHECK( cudaEventSynchronize(stop) );
    float innerTime;
    CUDA_CHECK( cudaEventElapsedTime(&innerTime, start, stop) );    

    CUDA_CHECK( cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost) );
    res = compare_results(gold, h_odata, size_x*size_y);
    if (res == false) {
      printf("*** %s kernel FAILED ***\n", kernelName);
      success = false;
    }
    
    // report effective bandwidths
    float innerBandwidth = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(innerTime/NUM_REPS);

    printf("transpose-Inner-%s, Throughput = %9.4f GB/s, Time = %.5f s, Size = %u fp32 elements\n", 
           kernelName, innerBandwidth, innerTime/NUM_REPS, (size_x * size_y));
#endif
   }
  
  printf("\n%s\n\n", (success == true) ? "PASSED" : "FAILED");

  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  CUDA_CHECK( cudaFree(d_idata) );
  CUDA_CHECK( cudaFree(d_odata) );

  CUDA_CHECK( cudaEventDestroy(start) );
  CUDA_CHECK( cudaEventDestroy(stop) );
  
  cudaThreadExit();
  return 0;

}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DIM 8

int printMat(float *mat, int numRow, int numCol, int mype, int nproc){

    float *buf;
    int i, j, k;
    MPI_Status status;

    if (mype == 0)
    {   
	for (i=0; i<numRow; i++)
	{
	    for (j=0; j<numCol; j++)
		printf("%.2f\t", mat[(i*numCol)+j]);
	    printf("\n");
	} 

	buf = (float *)malloc(numCol*numRow*sizeof(float));

	for (k=1; k < nproc; k++)
	{
	    MPI_Recv(buf, numRow*numCol, MPI_FLOAT, k, 0, MPI_COMM_WORLD, &status);
	    for (i=0; i < numRow; i++)
	    {
		for (j=0; j<numCol; j++)
		    printf("%.2f\t", buf[(i*numCol)+j]);
		printf("\n");
	    } 

	}

	printf("\n\n\n");
    }
    else
	MPI_Send(mat, numRow*numCol, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    return 0;
}



int main (int argc, char *argv[])
{
  FILE *fp;
  int count = 0, ind;
  float *A, *B, *C, *bufRecv, *bufSend, somma;
  int numRow, numCol, startInd;
  int nproc, mype; 
  int i, j, y, k, sup, r;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  /* Get local matrix dimensions */
  numRow = DIM / nproc;
  numCol = DIM;
  /* Avoid remainders */
  if (r = DIM % nproc)
    {
      if (mype == 0) printf("The number of process must divide %d exactly.\n",DIM);
      MPI_Finalize();
      return -1;      
    }

  /* Allocate workspace */
  A = (float *)malloc(numCol*numRow*sizeof(float));
  B = (float *)malloc(numCol*numRow*sizeof(float));
  C = (float *)malloc(numCol*numRow*sizeof(float));

  bufRecv = (float *)malloc(numCol*numRow*sizeof(float));
  bufSend = (float *)malloc(numRow*numRow*sizeof(float));

  /* Initialize matrices */
  for(i=0;i<numRow;++i)
  {
    for(j=0;j<numCol;++j)
      {
	A[(i*numCol)+j] = (((numRow * mype) + (i+1)) * (j+1));
	B[(i*numCol)+j] = 1 / A[(i*numCol)+j];
	C[(i*numCol)+j] = 0;
      }
  }
  /* Perform multiplication */
  for(count=0;count<nproc;++count)
    {
      startInd = count * numRow;
      for(i=0;i<numRow;++i)
	for(j=0;j<numRow;j++)
	  {
	    bufSend[j+(i*numRow)] = B[(i*numCol)+startInd+j];
	  }

      MPI_Allgather(bufSend, numRow*numRow, MPI_FLOAT, bufRecv, numRow*numRow, MPI_FLOAT, MPI_COMM_WORLD);
      for(k=0;k<numRow;++k){
	for(i=0, somma=0.0;i<numRow;++i)
	  {
	    somma=0.0;
	    for(j=0;j<numCol;++j)
	      {
		somma += A[(k*numCol)+j] * bufRecv[(numRow*j)+i];
	      }
	    C[(count*numRow)+(k*numCol)+i] = somma;
	  }
      }

    }

  /* Print matrices */
  printMat(A, numRow, numCol, mype, nproc);
  printMat(B, numRow, numCol, mype, nproc);
  printMat(C, numRow, numCol, mype, nproc); 

  /* Free workspace */
  free(A);
  free(B);
  free(C);
  free(bufSend);
  free(bufRecv);

  MPI_Finalize();
  return 0;

}                        
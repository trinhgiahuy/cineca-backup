#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10

typedef int row[N];

int main(int argc, char *argv[])
{
  int np, rank, n, r;
  int I;
  int i,j;
  int k;
  row *mat;
  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&np);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /* Number of rows that are assigned to each processor, taking care of the remainder */
  n = N / np;
  r = N % np;
  if (rank < r)
    n++;
  /* Allocate local workspace */
  mat = (row *) malloc(n * sizeof(row));
  /* Column of the first "one" entry */
  I = n*rank;
  if (rank >= r)
    I += r;
  /* Initilize local matrix */
  for (i=0; i < n; i++)
  {
    for (j=0; j<N; j++)
      if (j == I)
        mat[i][j] = 1;
      else
        mat[i][j] = 0;
    I++;
  }

  /* Print matrix */
  if (rank == 0)
    {
      /* Rank 0: print local buffer */
      for (i=0; i<n; i++)
    {
      for (j=0; j<N; j++)
        printf("%d ", mat[i][j]);
      printf("\n");
    }
      /* Receive new data from other processes 
     in an ordered fashion and print the buffer */
      for (k=1; k < np; k++)
    {
      if(k==r){
        n = n -1;
      }
      MPI_Recv(mat, n*N, MPI_INT, k, 0, MPI_COMM_WORLD, &status);
      for (i=0; i < n; i++)
        {
          for (j=0; j<N; j++)
        printf("%d ", mat[i][j]);
          printf("\n");
        }

    }
    }
  else
    {
      /* Send local data to Rank 0 */
      MPI_Send(mat, n*N, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

  free(mat);  
  MPI_Finalize();
  return 0;
}
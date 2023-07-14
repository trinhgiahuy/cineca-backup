#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10

/* This defines a type called "row" that is actually an array of N integers */
typedef int row[N];

int main(int argc, char *argv[])
{
  int np, rank, n, r;
  int I;
  int i,j;
  int k;
  row *mat;
  MPI_Status status;

/* Initialize MPI environment and get size and rank in "np" and "rank" respectively */
...

  /* Number of rows that are assigned to each processor, taking care of the remainder */
  n = N / np;
  r = N % np;
  if (rank < r)
    n++;

  /* Allocate local workspace, as an array of contiguous elements */
  mat = (row *) malloc(n * sizeof(row));
  /* Column of the first "one" entry. It sets up a starting position from which every rank should check the elements of their local matrix */
  I = n*rank;
  if (rank >= r)
    I += r;

  /* Initialize local matrix */
  for (i=0; i < n; i++)
  {
    for (j=0; j<N; j++)
      if (j == I)
        mat[i][j] = 1;
      else
        mat[i][j] = 0;
    I++;   /* Why? */
  }

  /* Print matrix */
  /* Set up the "if" condition so that rank 0 receives from everyone and prints the final matrix, and the other ranks are sending their local matrix to it. */
  if (...)
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
      /* This is the receiver task. Set up the receive call */
      ...
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
      ...
    }

  free(mat);  
  /* Finalize the environment */
 ...
  return 0;
}

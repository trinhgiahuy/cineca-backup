/* 
 * This program demonstrates the use of MPI_Alltoall when
 * transposing a square matrix.
 * For simplicity, the number of processes is 4 and the dimension
 * of the matrix is fixed to NROW
 */

#include <stdio.h>
#include <mpi.h>
#define NROW 16384
#define NP 4
#define NBLK NROW/NP
#define ONEML 99999999999
#define SIZE NROW*NROW*2*8
void
trans (double *a, int n)
/* transpose square matrix a, dimension nxn
 * Consider this as a black box for the MPI course
 */

{
  int i, j;
  int ij, ji, l;
  double tmp;
  ij = 0;
  l = -1;
  for (i = 0; i < n; i++)
    {
      l += n + 1;
      ji = l;
      ij += i + 1;
      for (j = i+1; j < n; j++)
	{
	  tmp = a[ij];
	  a[ij] = a[ji];
	  a[ji] = tmp;
	  ij++;
	  ji += n;
	}
    }
}

int
main (int argc, char *argv[])
{
  double a[NROW][NBLK];
  double b[NROW][NBLK];

  int i, j, nprocs, rank,provided,itemp,omp_get_num_threads(void);
  double r0,r1;

  MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided);
  //MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  if(rank == 0)
  {
    printf ("MPI thread support required and provided: %d %d\n",MPI_THREAD_FUNNELED,provided);
    printf("Transposing a %d x %d matrix, divided among %d processors\n",NROW,NROW,NP);
#pragma omp parallel
#pragma omp master
    printf("Using %d threads\n",omp_get_num_threads());
  }
  if (nprocs != NP)
    {
      if (rank == 0)
	printf ("Error, number of processes must be %d\n",NP);
      MPI_Finalize ();
      return 1;
    }
#pragma omp parallel 
{  
#pragma omp master
 r0 = MPI_Wtime();
#pragma omp for private(i,j)
   for (i = 0; i < NROW; i++)
    for (j = 0; j < NBLK; j++)
      a[i][j] = ONEML * i + j + NBLK * rank; /* give every element a unique value */

#pragma omp master
  {
  r0 = MPI_Wtime()-r0; 
  {
  if(rank == 0)
  {
    printf("Building matrix time (sec) %f\n",r0);
  }
  }
  /* do the MPI part of the transpose */
  /* Tricky here is the number of items to send and receive. 
   * Not NROWxNBLK as one may guess, but the amount to send to one process
   * and the amount to receive from any process */
  /* MPI_Alltoall does not a transpose of the data received, we have to
   * do this ourself: */
 r0 = MPI_Wtime();
  MPI_Alltoall (&a[0][0],	/* address of data to send  */
		NBLK * NBLK,	/* number of items to send to one process  */
		MPI_DOUBLE,	/* type of data  */
		&b[0][0],	/* address for receiving the data  */
		/* NOTE: send data and receive data may NOT overlap */
		NBLK * NBLK,	/* number of items to receive 
				   from any process  */
		MPI_DOUBLE,	/* type of receive data  */
		MPI_COMM_WORLD);
 r0 = MPI_Wtime()-r0;
  if(rank == 0)
  {
    printf("MPI_Alltoall time (sec) %f\n",r0);
  }
 }
#pragma omp barrier
  /* transpose NP square matrices, order NBLKxNBLK: */
#pragma omp master
r1 = MPI_Wtime();
#pragma omp for
  for (i = 0; i < NP; i++)
      trans (&b[i * NBLK][0], NBLK);
#pragma omp master
r1 = MPI_Wtime()-r1;
 }
  if(rank == 0)
  {
    printf("Transpose block matrices time (sec) %f\n",r1);
    printf("Bandwidth (GB/sec) %f\n",(double)SIZE*1.e-9/(r1+r0));
  }

   /* now check the result */

  for (i = 0; i < NROW; i++)
    for (j = 0; j < NBLK; j++)
      {
	if (b[i][j] != ONEML * (j + NBLK * rank) + i )
	  {
	    printf ("process %d found b[%d][%d] = %f, but %f was expected\n",
		    rank, i, j, b[i][j], (double) (ONEML * (j + NBLK * rank) + i));
	    MPI_Abort (MPI_COMM_WORLD,1);
	    return 1;
	  }
      }
  if (rank == 0)
    printf ("Transpose seems ok\n");
  MPI_Finalize ();
  return 0;
}
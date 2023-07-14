#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

/* #define ndata 1000 */
#define ndata 1000000


int main(int argc, char* argv[]){

    int me, nprocs, i=0, left, right;
    MPI_Status status;
    MPI_Request request; 

    float a[ndata];
    float b[ndata];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    /* Initialize data */
    for(i=0;i<ndata;++i)
      {
    a[i] = me;
    b[i] = -1;
      }
    /* Compute neighbour ranks. By using the module operation, we can guarantee periodicity */
    right = (me+1)%nprocs;
    left  = (me-1+nprocs)%nprocs;
    
    /* Sendrecv data */
    MPI_Isend(a, ndata, MPI_FLOAT, right, 0, MPI_COMM_WORLD, &request);
    MPI_Recv(b, ndata, MPI_FLOAT, left, 0, MPI_COMM_WORLD, &status);

    MPI_Wait(&request,&status);

    printf("\tI am task %d and I have received b(0) = %1.2f \n", me, b[0]);

    MPI_Finalize();
    return 0;
}

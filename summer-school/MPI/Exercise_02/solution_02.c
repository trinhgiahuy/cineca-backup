#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define NDATA 10000

int main(int argc, char* argv[]){

    int me, nprocs, i = 0;
    MPI_Status status;

    float a[NDATA];
    float b[NDATA];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    /* Initialize data */
    for(i=0;i<NDATA;++i){
    a[i] = me;
    }
    
    /* Protect against the use with a number of processes != 2 */
    if(nprocs!=2){
      if(me==0)
        printf("\n\tThis program must run on 2 processors");
      MPI_Finalize();
      return 0;
    }

    /* Send and Receive data */
    if(me==0){
      MPI_Send(a, NDATA, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(b, NDATA, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, &status);
    }
    else{
      MPI_Recv(b, NDATA, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Send(a, NDATA, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

    printf("\tI am task %d and I have received b(0) = %1.2f \n", me, b[0]);

    MPI_Finalize();
    return 0;
}

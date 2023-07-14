#include <stdio.h>
#include <mpi.h>

int main (int argc, char* argv[]) {

    int rank, size;
    /* Declare a new MPI Datatype named "diag" */
    ...

    /* MPI Initialization */
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    /* Matrix initialization */
    int matrix[size][size];

    for (int i=0; i<size; i++) 
	for (int j=0; j<size; j++)
	    if (i==j) matrix[i][j] = rank;
 	    else matrix[i][j] = 0;

    /* Print rank 0 matrix (should be filled with 0s) */
    if (rank == 0) {
    printf("Rank 0 matrix before communication:\n");
       for (int i=0; i<size; i++) {
           for (int j=0; j<size; j++)
               printf("%d ",matrix[i][j]);
           printf("\n");
           }
       printf("\n"); 
       }

    /* Diagonal datatype vector creation and commitment. What is the blocklength and the displacement? */
    ...
    ...

    /* Communication: rank 0 gathers all the diagonals from the other ranks and stores them in the row corresponding to the
     * sending rank. Note that 1 "diag" type is sent and size MPI_INT types are received, so that the values can be stored
     * contiguously in the receiving matrix */
    MPI_Gather(...);

    /* Print rank 0 matrix after communication (each element should be its row number) */
    if (rank == 0) {
    printf("Rank 0 matrix after communication:\n");
       for (int i=0; i<size; i++) {
           for (int j=0; j<size; j++)
               printf("%d ",matrix[i][j]);
           printf("\n");
           }
       printf("\n");
       }

    /* Remember to free the datatype! */
    ...

    MPI_Finalize();
    return 0;
    }

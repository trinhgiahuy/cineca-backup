#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) { 

// Declare variables (or do it later)
 int n_proc, n_rank, i,j ;
 const int n=5, nb=2;
 float a[n][n];

 /* We need to declare an MPI handle for the new datatype. What type will it be? */
 ... myvector;
 MPI_Status mystatus;

// Start MPI
 MPI_Init(&argc,&argv);
 MPI_Comm_size(MPI_COMM_WORLD,&n_proc);
 MPI_Comm_rank(MPI_COMM_WORLD,&n_rank);

// Check if the number of processes is 2
 if(n_proc != 2) {
    if(n_rank == 0) printf("Test program has to work only with two MPI processes\n");
   MPI_Finalize();
   exit(1);
 }

// Initialize matrix
 if(n_rank == 0) for(i=0;i<n;i++) for(j=0;j<n;j++) a[i][j] = 0.F;
 if(n_rank == 1) for(i=0;i<n;i++) for(j=0;j<n;j++) a[i][j] = 1.F;

// Define vector and commit the new datatype in "myvector"
 ...
 ...
 
 // Print matrix a for rank=1
 if(n_rank == 1) {
    printf("Matrix A before communications:\n");
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
 }

// Communicate. Remember that we are sending one istance of the now datatype "myvector"
 if(n_rank == 0) {
    MPI_Send(...);
 }
 if(n_rank == 1) {
    MPI_Recv(...);
 }

// Print matrix a for rank=1
 if(n_rank == 1) {
    printf("Matrix A after communications:\n");
    for(i=0;i<n;i++) {
        for(j=0;j<n;j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
 }

// Free the allocated vector datatype. How?
 ...

// Finalize MPI
 MPI_Finalize();

 return 0;
}

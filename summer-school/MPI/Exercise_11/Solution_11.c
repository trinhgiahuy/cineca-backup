
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char **argv) { 

// Declare variables (or do it later)
 int n_proc, n_rank, i,j ;
 MPI_Datatype myvector;
 const int n=5, nb=2;
 float a[n][n];
 MPI_Status mystatus;

// Start MPI
 MPI_Init(&argc,&argv);
 MPI_Comm_size(MPI_COMM_WORLD,&n_proc);
 MPI_Comm_rank(MPI_COMM_WORLD,&n_rank);

// Check the number of processes is 2
 if(n_proc != 2) {
    if(n_rank == 0) printf("Test program has to work only with two MPI processes\n");
   MPI_Finalize();
   exit(1);
 }

// Initialize matrix
 if(n_rank == 0) for(i=0;i<n;i++) for(j=0;j<n;j++) a[i][j] = 0.F;
 if(n_rank == 1) for(i=0;i<n;i++) for(j=0;j<n;j++) a[i][j] = 1.F;

// Define vector
 MPI_Type_vector(n, nb, n, MPI_FLOAT, &myvector);
 MPI_Type_commit(&myvector);

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

// Communicate
 if(n_rank == 0) {
    MPI_Send(a, 1, myvector, 1, 100, MPI_COMM_WORLD);
 }
 if(n_rank == 1) {
    MPI_Recv(a, 1, myvector, 0, 100, MPI_COMM_WORLD, &mystatus);
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

 MPI_Type_free(&myvector);

// Finalize MPI
 MPI_Finalize();

 return 0;
}

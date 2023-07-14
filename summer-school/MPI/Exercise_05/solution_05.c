  #include <stdlib.h>
  #include <stdio.h>
  #include <mpi.h>

  #define N 20

  int main(int argc, char* argv[]){

    int my_rank, nprocs;
    int i,j;
    int rem, num_local_row;
    int *matrix;
    int proc_up, proc_down;
    MPI_Status status1, status2;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    /*  numer of rows for each mpi task */
    rem= N % nprocs;
    num_local_row = (N - rem)/nprocs;

    if(my_rank < rem) num_local_row = num_local_row+1;

    /* matrix allocation - as a single array so the memory stays contiguous */
    matrix = (int*) malloc(((num_local_row+2)*N)*sizeof(int));

    /* initialization of the local matrix */
    for(i=0; i<num_local_row+2 ;i++){
      for(j=0; j<N; j++){
        matrix[i*N+j] = my_rank ;
      }
    }
    proc_down = my_rank+1;
    proc_up = my_rank-1;
    if(proc_down==nprocs) proc_down=0;
    if(proc_up < 0) proc_up=nprocs-1;

    /* check printings */
    /* printf("rank %d, proc up %d, proc down %d\n",
    my_rank, proc_up, proc_down); */

    /* printf("my_rank %d, num_local_row %d\n", my_rank, num_local_row); */

    /* printf("my_rank %d,matrix[0] %d, matrix[N] %d,
    matrix[(N)*(num_local_row+2)] %d \n", my_id, matrix[0],
    matrix[N], matrix[N*(num_local_row+2)]); */

    /* send receive of the ghost reagions */
    MPI_Sendrecv(&matrix[N],N,MPI_INTEGER,proc_up,10,
      &matrix[N*((num_local_row+1))],N,MPI_INTEGER,proc_down,10,
      MPI_COMM_WORLD,&status1);

    MPI_Sendrecv(&matrix[N*num_local_row],N,MPI_INTEGER,proc_down,
      11,&matrix[0],N,MPI_INTEGER,proc_up,11,MPI_COMM_WORLD,&status2);


    /* check printings */
    printf("\n ");

    printf("my_rank, %d, riga arrivata da sopra: ", my_rank);
    for(i=0;i<N;i++){
      printf("%d \t", matrix[i]);
    }
    printf("\n ");

    printf("my_rank, %d, riga arrivata da sotto: ", my_rank);
    for(i=N*(num_local_row+1);i<N*(num_local_row+1)+N;i++){
      printf("%d \t", matrix[i]);
    }
    printf("\n ");

    free(matrix);
    MPI_Finalize();
    return 0;
  }

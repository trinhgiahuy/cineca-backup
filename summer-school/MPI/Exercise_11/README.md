# Exercise 11

Write a program working only with 2 MPI processes.

For each process, define a square matrix A (nXn). Rank 0 fills the matrix with 0, while rank 1 fills it with 1. Define a datatype that handles a column (if C) or a row (If Fortran) of A.

Now begin the communication: rank 0 sends the first column/row of A to rank 1, overwriting its own first column/row. Check the results by printing the matrix on the screen. <br> Modify the code by sending the first nb columns/rows of A: do you have to change the type? Can you send two items of the new type?

![alt text](../images/es11.png)



## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_TYPE_VECTOR](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_vector.3.php) | int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype \*newtype) | MPI_TYPE_VECTOR(COUNT, BLOCKLENGTH, STRIDE, OLDTYPE, NEWTYPE, IERROR) <br> INTEGER COUNT, BLOCKLENGTH, STRIDE, OLDTYPE, NEWTYPE, IERROR |
| [MPI_TYPE_COMMIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_commit.3.php) | int MPI_Type_commit(MPI_Datatype \*datatype) | MPI_TYPE_COMMIT(DATATYPE, IERROR) <br> INTEGER DATATYPE, IERROR |
| [MPI_TYPE_FREE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_free.3.php) | int MPI_Type_free(MPI_Datatype \*datatype) | MPI_TYPE_FREE(DATATYPE, IERROR) <br> INTEGER DATATYPE, IERROR |
| [MPI_TYPE_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_size.3.php) | int MPI_Type_size(MPI_Datatype \*datatype, int \*size) | MPI_TYPE_SIZE(DATATYPE, SIZE, IERROR) <br> INTEGER DATATYPE, SIZE, IERROR |
| [MPI_TYPE_GET_EXTENT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_get_extent.3.php) | int MPI_Type_get_extent(MPI_Datatype \*datatype, MPI_Aint \*lb, MPI_Aint \*extent) | MPI_TYPE_GET_EXTENT(DATATYPE, LB, EXTENT, IERROR) <br> INTEGER DATATYPE, IERROR <br> INTEGER (KIND=MPI_ADDRESS_KIND) LB, EXTENT |
| [MPI_SEND](https://www.open-mpi.org/doc/v3.1/man3/MPI_Send.3.php) | int MPI_Send(void\* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) | MPI_SEND(BUF, COUNT, DATATYPE, DEST, TAG, COMM, IERROR) <br> \<type\> BUF(\*) INTEGER COUNT, DATATYPE, DEST, TAG, COMM, IERROR |
| [MPI_RECV](https://www.open-mpi.org/doc/v3.1/man3/MPI_Recv.3.php) | int MPI_Recv(void\* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status \*status) | MPI_RECV(BUF, COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS, IERROR) <br> \<type\> BUF(\*) <br> INTEGER COUNT, DATATYPE, SOURCE, TAG, COMM, STATUS(MPI_STATUS_SIZE), IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |

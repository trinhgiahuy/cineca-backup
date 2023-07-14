# Exercise 9

Each process initializes a one-dimensional array by giving to all the elements the value of its rank+1. Then the root process (task 0) performs a sum reduce operation to the arrays of all the processes, and after that performs a product reduce on the same arrays. Finally, each process generates a random number and root process finds (and prints) the maximum value among these random values.

Modify the code to perform a simple scalability test using MPI_Wtime. This function/routine returns a floating-point number of seconds, representing elapsed wall-clock time since some time in the past. You can call it at the beginning and at the end of your MPI part of the code, then use the difference so store the elapsed time. Notice what happens when you go up with the number of processes involved.

## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_REDUCE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Reduce.3.php) | int MPI_Reduce(void\* sendbuf, void\* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) | MPI_REDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) <br> \<type> SENDBUF(\*), RECVBUF(\*) INTEGER COUNT, DATATYPE, OP, ROOT, COMM, IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |
| [MPI_WTIME](https://www.open-mpi.org/doc/v3.1/man3/MPI_Wtime.3.php) | double MPI_Wtime(void) | MPI_WTIME() |

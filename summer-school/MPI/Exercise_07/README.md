# Exercise 7

Task 0 initializes a variable to a given value,then modifies the variable (for example, by calculating the square of its value) and finally broadcasts it to all the others tasks.

## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_BCAST](https://www.open-mpi.org/doc/v3.1/man3/MPI_Bcast.3.php) | int MPI_Bcast(void\* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) | MPI_BCAST(BUFFER, COUNT, DATATYPE, ROOT, COMM, IERROR) <br> \<type\> BUFFER(\*) <br> INTEGER COUNT, DATATYPE, ROOT, COMM, IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |

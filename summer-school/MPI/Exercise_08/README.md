# Exercise 8

Task 0 initializes a one-dimensional array assigning to each cell the value of its index. This array is then divided into chunks and sent to other processes. After having received the proper chunk, each process updates it by adding its rank and then sends it back to root process. (Analyze the cases for equal and not equal chunks separately).

## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_SCATTER](https://www.open-mpi.org/doc/v3.1/man3/MPI_Scatter.3.php) | int MPI_Scatter(void\* sendbuf, int sendcount, MPI_Datatype sendtype, void\* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) | MPI_SCATTER(SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) <br> \<type\> SENDBUF(\*), RECVBUF(\*) <br> INTEGER SENDCOUNT, SENDTYPE, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR |
| [MPI_GATHER](https://www.open-mpi.org/doc/v3.1/man3/MPI_Gather.3.php) | int MPI_Gather(void\* sendbuf, int sendcount, MPI_Datatype sendtype, void\* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) | MPI_GATHER(SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) <br> \<type\> SENDBUF(\*), RECVBUF(\*) <br> INTEGER SENDCOUNT, SENDTYPE, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR |
| [MPI_SCATTERV](https://www.open-mpi.org/doc/v3.1/man3/MPI_Scatterv.3.php) | int MPI_Scatterv(void\* sendbuf, int \*sendcounts, int \*displs, MPI_Datatype sendtype, void\* recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) | MPI_SCATTERV(SENDBUF, SENDCOUNTS, DISPLS, SENDTYPE, RECVBUF, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR) <br> \<type\> SENDBUF(\*), RECVBUF(\*) <br> INTEGER SENDCOUNTS(\*), DISPLS(\*), SENDTYPE, RECVCOUNT, RECVTYPE, ROOT, COMM, IERROR |
| [MPI_GATHERV](https://www.open-mpi.org/doc/v3.1/man3/MPI_Gatherv.3.php) | int MPI_Gatherv(void\* sendbuf, int sendcount, MPI_Datatype sendtype, void\* recvbuf, int \*recvcounts, int \*displs, MPI_Datatype recvtype, int root, MPI_Comm comm) | MPI_GATHERV(SENDBUF, SENDCOUNT, SENDTYPE, RECVBUF, RECVCOUNTS, DISPLS, RECVTYPE, ROOT, COMM, IERROR) <br> \<type\> SENDBUF(\*), RECVBUF(\*) <br> INTEGER SENDCOUNT, SENDTYPE, RECVCOUNTS(\*), DISPLS(\*), RECVTYPE, ROOT, COMM, IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |

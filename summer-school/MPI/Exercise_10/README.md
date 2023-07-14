# Exercise 10

Create a 2-dimensional cartesian grid topology to communicate between processes.

In each task is initialized a variable with the local rank of the cartesian communicator. The exercise is divided in three steps:

**1)** Compare the local rank with the global MPI_COMM_WORLD rank. Are they the same number?

**2)** Calculate on each task the average between its local rank and the local rank from each of its neighbours (north, east, south, west). Notice that in order to do this the cartesian communicator has to be periodic (the bottom rank is neighbour of the top).

**3)** Calculate the average of the local ranks on each row and column. Create a family of sub-cartesian communicators to allow the communications between rows and columns.

## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_DIMS_CREATE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Dims_create.3.php) | int MPI_Dims_create(int nnodes, int ndims, int \*dims) | MPI_DIMS_CREATE(NNODES, NDIMS, DIMS, IERROR) <br> INTEGER NNODES,NDIMS,DIMS(\*),IERROR |
| [MPI_CART_CREATE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_create.3.php) | int MPI_Cart_create(MPI_Comm comm_old, int ndims, int \*dims, int \*periods, int reorder, MPI_Comm \*comm_cart) | MPI_CART_CREATE(COMM_OLD, NDIMS, DIMS, PERIODS, REORDER, COMM_CART, IERROR) <br> INTEGER COMM_OLD, NDIMS, DIMS(\*), COMM_CART, IERROR <br> LOGICAL PERIODS(\*), REORDER |
| [MPI_CART_COORDS](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_coords.3.php) | int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int \*coords) | MPI_CART_COORDS(COMM, RANK, MAXDIMS, COORDS, IERROR) <br> INTEGER COMM, RANK, MAXDIMS, COORDS(\*), IERROR |
| [MPI_CART_SHIFT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_shift.3.php) | int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int \*rank_source, int \*rank_dest) | MPI_CART_SHIFT(COMM, DIRECTION, DISP, RANK_SOURCE, RANK_DEST, IERROR) <br> INTEGER COMM, DIRECTION, DISP, RANK_SOURCE, RANK_DEST, IERROR |
| [MPI_CART_SUB](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_sub.3.php) | int MPI_Cart_sub(MPI_Comm comm, int \*remain_dims, MPI_Comm \*newcomm) | MPI_CART_SUB(COMM, REMAIN_DIMS, NEWCOMM, IERROR) <br> INTEGER COMM, NEWCOMM, IERROR <br> LOGICAL REMAIN_DIMS(\*) |
| [MPI_COMM_FREE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_free.3.php) | int MPI_Comm_free(MPI_Comm \*comm) | MPI_COMM_FREE(COMM, IERROR) <br> INTEGER COMM, IERROR |
| [MPI_SENDRECV](https://www.open-mpi.org/doc/v3.1/man3/MPI_Sendrecv.3.php) | int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status) | MPI_SENDRECV(SENDBUF, SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVBUF, RECVCOUNT, RECVTYPE, SOURCE, RECVTAG, COMM, STATUS, IERROR) <br> \<type\> SENDBUF(\*), RECVBUF(\*) <br> INTEGER SENDCOUNT, SENDTYPE, DEST, SENDTAG, RECVCOUNT, RECVTYPE, SOURCE, RECV TAG, COMM, STATUS(MPI_STATUS_SIZE), IERROR |
| [MPI_REDUCE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Reduce.3.php) | int MPI_Reduce(void\* sendbuf, void\* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) | MPI_REDUCE(SENDBUF, RECVBUF, COUNT, DATATYPE, OP, ROOT, COMM, IERROR) <br> \<type> SENDBUF(\*), RECVBUF(\*) INTEGER COUNT, DATATYPE, OP, ROOT, COMM, IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |

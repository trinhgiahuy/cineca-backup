# Exercise 14

Write a program which decomposes an integer matrix (m x n) using a 2D MPI cartesian grid:
- handle the remainders for non multiple sizes
- fill the matrix with the row-linearized indexes

#### Aij = m · i + j

- reconstruct the absolute indexes from the local ones
- remember that in C the indexes of arrays start from 0

![alt text](../images/es14.png)

The program writes the matrix to file using MPI-I/O collective write and using MPI data-types:
- which data-type do you have to use?

Check the results using:
- shell Command : 

```
od -i output.dat
```

- parallel MPI-I/O read functions (similar to write structure)
- serial standard C and Fortran check: <br>
    • only rank=0 performs check <br>
    • read row-by-row in C and column-by-column in Fortran and check each element of the row/columns <br>
    • use binary files and fread in C <br>
    • use unformatted and access='stream' in Fortran

Which check is the most scrupolous one?
- is the Parallel MPI-I/O check enough?


## HINTS:

|    | **C** | **FORTRAN** |
|----|-------|-------------|
| [MPI_FILE_OPEN](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_open.3.php) | int MPI_File_open(MPI_Comm comm, char \*filename, int amode, MPI_Info info, MPI_File \*FH) | MPI_FILE_OPEN(COMM, FILENAME, AMODE, INFO, FH, IERROR) <br> CHARACTER\*(\*) FILENAME <br> INTEGER COMM, AMODE, INFO, FH, IERROR |
| [MPI_FILE_DELETE](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_delete.3.php) | int MPI_File_delete(char \*filename, MPI_Info info) | MPI_FILE_DELETE(FILENAME, INFO, IERROR) <br> CHARACTER\*(\*) FILENAME <br> INTEGER INFO, IERROR |
| [MPI_FILE_CLOSE](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_close.3.php) | int MPI_File_close(MPI_File \*fh) | MPI_FILE_CLOSE(FH, IERROR) <br> INTEGER FH, IERROR |
| [MPI_FILE_SET_VIEW](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_set_view.3.php) | int MPI_File_set_view(MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, char \*datarep, MPI_Info info) | MPI_FILE_SET_VIEW(FH, DISP, ETYPE, FILETYPE, DATAREP, INFO, IERROR) <br> CHARACTER\*(\*) DATAREP <br> INTEGER(KIND=MPI_OFFSET_KIND) DISP <br> INTEGER FH, ETYPE, FILETYPE, INFO, IERROR |
| [MPI_FILE_READ_ALL](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_read_all.3.php) | int MPI_File_read_all(MPI_File fh, void \*buf, int count, MPI_Datatype datatype, MPI_Status \*status) | MPI_FILE_READ_ALL(FH, BUF, COUNT, DATATYPE, STATUS, IERROR) <br> <type> BUF(\*) <br> INTEGER FH, COUNT, DATATYPE, STATUS(MPI_STATUS_SIZE), IERROR |
| [MPI_FILE_WRITE_ALL](https://www.open-mpi.org/doc/v3.1/man3/MPI_File_write_all.3.php) | int MPI_File_write_all(MPI_File fh, void \*buf, int count, MPI_Datatype datatype, MPI_Status \*status) | MPI_FILE_WRITE_ALL(FH, BUF, COUNT, DATATYPE, STATUS, IERROR) <br> <type> BUF(\*) <br> INTEGER FH, COUNT, DATATYPE, STATUS(MPI_STATUS_SIZE), IERROR |
| [MPI_TYPE_CEATE_SUBARRAY](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_create_subarray.3.php) | int MPI_Type_create_subarray(int ndims, int array_of_sizes[], int array_of_subsizes[], int array_of_starts[], int order, MPI_Datatype oldtype, MPI_Datatype \*newtype) | MPI_TYPE_CREATE_SUBARRAY(NDIMS, ARRAY_OF_SIZES, ARRAY_OF_SUBSIZES, ORDER, OLDTYPE, NEWTYPE, IERROR) <br> INTEGER NDIMS, ARRAY_OF_SIZES(\*), ARRAY_OF_SUBSIZES(\*), ARRAY_OF_STARTS(\*), ORDER, OLDTYPE, NEWTYPE, IERROR |
| [MPI_TYPE_COMMIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_commit.3.php) | int MPI_Type_commit(MPI_Datatype \*datatype) | MPI_TYPE_COMMIT(DATATYPE, IERROR) <br> INTEGER DATATYPE, IERROR |
| [MPI_TYPE_FREE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Type_free.3.php) | int MPI_Type_free(MPI_Datatype \*datatype) | MPI_TYPE_FREE(DATATYPE, IERROR) <br> INTEGER DATATYPE, IERROR |
| [MPI_DIMS_CREATE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Dims_create.3.php) | int MPI_Dims_create(int nnodes, int ndims, int \*dims) | MPI_DIMS_CREATE(NNODES, NDIMS, DIMS, IERROR) <br> INTEGER NNODES,NDIMS,DIMS(\*),IERROR |
| [MPI_CART_CREATE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_create.3.php) | int MPI_Cart_create(MPI_Comm comm_old, int ndims, int \*dims, int \*periods, int reorder, MPI_Comm \*comm_cart) | MPI_CART_CREATE(COMM_OLD, NDIMS, DIMS, PERIODS, REORDER, COMM_CART, IERROR) <br> INTEGER COMM_OLD, NDIMS, DIMS(\*), COMM_CART, IERROR <br> LOGICAL PERIODS(\*), REORDER |
| [MPI_CART_COORDS](https://www.open-mpi.org/doc/v3.1/man3/MPI_Cart_coords.3.php) | int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int \*coords) | MPI_CART_COORDS(COMM, RANK, MAXDIMS, COORDS, IERROR) <br> INTEGER COMM, RANK, MAXDIMS, COORDS(\*), IERROR |
| [MPI_COMM_FREE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_free.3.php) | int MPI_Comm_free(MPI_Comm \*comm) | MPI_COMM_FREE(COMM, IERROR) <br> INTEGER COMM, IERROR |
| [MPI_INIT](https://www.open-mpi.org/doc/v3.1/man3/MPI_Init.3.php) | int MPI_Init(int \*argc, char \***argv) | MPI_INIT(IERROR) <br> INTEGER IERROR |
| [MPI_COMM_SIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_size.3.php) | int MPI_Comm_size(MPI_Comm comm, int \*size) | MPI_COMM_SIZE(COMM, SIZE, IERROR) <br> INTEGER COMM, SIZE, IERROR |
| [MPI_COMM_RANK](https://www.open-mpi.org/doc/v3.1/man3/MPI_Comm_rank.3.php) | int MPI_Comm_rank(MPI_Comm comm, int \*rank) | MPI_COMM_RANK(COMM, RANK, IERROR) <br> INTEGER COMM, RANK, IERROR |
| [MPI_FINALIZE](https://www.open-mpi.org/doc/v3.1/man3/MPI_Finalize.3.php) | int MPI_Finalize(void) | MPI_FINALIZE(IERROR) <br> INTEGER IERROR |
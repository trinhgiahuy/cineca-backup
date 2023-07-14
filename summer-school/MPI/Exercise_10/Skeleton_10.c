#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

int main(int argc, char *argv[])
{

  int world_rank, row_rank, col_rank, cart_rank;  /* Why do we need all of these? */
  int nprocs, row_size, col_size;
  int coords[2], sub_coords[2];
  int dims[2] = {0, 0}, period[2] = {1, 1};  /* What do these initializations mean? */
  int src_rank, dst_rank;
  int sum, temp;
  float avg;

  /* Declare MPI variables: we need three new communicators that we will call cart_grid (main cartesian), cart_row (local row) and cart_col (local column). What is their type? */
  ...
  MPI_Status status;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* Cartesian grid creation. Use MPI_Dims_create for the selection of the proper dimension array and create the cartesian communicator, to be stored in cart_grid */
  MPI_Dims_create(...);
  MPI_Cart_create(...);

  /*Local cart_rank initialization and comparison to global world_rank. Use MPI_Comm_rank as usual, but what to put instead of MPI_COMM_WORLD? */
  MPI_Comm_rank(...);

  printf ("I am world_rank %d in MPI_COMM_WORLD and world_rank %d in the cartesian communicator\n", world_rank, cart_rank);

  /* Coordinates creation and neighbour communication */
  MPI_Cart_coords(...);

  /* Communication south. Complete MPI_Cart_shift so that is coherent with MPI_Sendrecv below. South= direction y, looking down */
  sum = world_rank;
  MPI_Cart_shift(...);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /* Communication north= direction y, looking up */
  MPI_Cart_shift(...);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Communication east= direction x, looking right */
  MPI_Cart_shift(...);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Communication west= direction x, looking left */
  MPI_Cart_shift(...);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Neighbour's average */
  avg = (float) sum / 5;
  printf("Cart rank %d (%d, %d), neighbours average: %.2f\n", cart_rank, coords[0], coords[1], avg);

  /*Row sub-communicator creation. How to fill sub_coords so that the correct dimension is dropped? */
  sum = 0;
  sub_coords[0] = ...;
  sub_coords[1] = ...;
  MPI_Cart_sub(...);  /* Decide what is the starting communicator and the new communicator. Use that info for the two commands below */
  MPI_Comm_size(..., &row_size);
  MPI_Comm_rank(..., &row_rank);

  /*Row sub-communicator's average calculation. First, the sum of all local ranks (use the variable "sum" for this) */
  MPI_Reduce(...);

/* The sum calculated is divided for the number of elements in the row communicator */
  if (row_rank == 0) {
    avg = (float) sum / row_size;
    printf("Row %d, row average: %.2f\n", coords[0], avg);
  }

  /*Row sub-communicator creation. How to fill sub_coords so that the correct dimension is dropped? */
  sum = 0;
  sub_coords[0] = ...;
  sub_coords[1] = ...;
  MPI_Cart_sub(...);  /* Decide what is the starting communicator and the new communicator. Use that info for the two commands below */
  MPI_Comm_size(..., &col_size);
  MPI_Comm_rank(..., &col_rank);

  /* Column sub-communicator's average calculation. First, the sum of all local ranks (use the variable "sum" for this) */
  MPI_Reduce(...);

/* The sum calculated is divided for the number of elements in the column communicator */
  if (col_rank == 0) {
    avg = (float) sum / col_size;
    printf("Column %d, column average: %.2f\n", coords[1], avg);
  }

  /* Finalization operations. Every communicator created must be freed: what is the command for this? */
  MPI_C...;
  MPI_C...;
  MPI_C...;
  MPI_Finalize();

  return 0;
}

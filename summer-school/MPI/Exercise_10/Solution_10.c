#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{

  int world_rank, row_rank, col_rank, cart_rank;
  int nprocs, row_size, col_size;
  int coords[2], sub_coords[2];
  int dims[2] = {0, 0}, period[2] = {1, 1};
  int src_rank, dst_rank;
  int sum, temp;
  float avg;

  MPI_Comm cart_grid, cart_row, cart_col;
  MPI_Status status;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  /* Cartesian grid creation */
  MPI_Dims_create(nprocs, 2, dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, period, 1, &cart_grid);

  /*Local world_rank initialization and comparison to global world_rank */
  MPI_Comm_rank(cart_grid, &cart_rank);

  printf ("I am world_rank %d in MPI_COMM_WORLD and world_rank %d in the cartesian communicator\n", world_rank, cart_rank);

  /* Coordinates creation and neighbour communication */
  MPI_Cart_coords(cart_grid, cart_rank, 2, coords);

  /* Communication south */
  sum = world_rank;
  MPI_Cart_shift(cart_grid, 1, 1, &src_rank, &dst_rank);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /* Communication north */
  MPI_Cart_shift(cart_grid, 1, -1, &src_rank, &dst_rank);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Communication east */
  MPI_Cart_shift(cart_grid, 0, 1, &src_rank, &dst_rank);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Communication west */
  MPI_Cart_shift(cart_grid, 0, -1, &src_rank, &dst_rank);
  MPI_Sendrecv(&world_rank, 1, MPI_INT, dst_rank, 0, &temp, 1, MPI_INT, src_rank, 0, cart_grid, &status);
  sum += temp;

  /*Neighbour's average */
  avg = (float) sum / 5;
  printf("Cart rank %d (%d, %d), neighbours average: %.2f\n", cart_rank, coords[0], coords[1], avg);

  /*Row sub-communicator creation */
  sum = 0;
  sub_coords[0] = 0;
  sub_coords[1] = 1;
  MPI_Cart_sub(cart_grid, sub_coords, &cart_row);
  MPI_Comm_size(cart_row, &row_size);
  MPI_Comm_rank(cart_row, &row_rank);

  /*Row sub-communicator's average calculation */
  MPI_Reduce(&world_rank, &sum, 1, MPI_INT, MPI_SUM, 0, cart_row);

  if (row_rank == 0) {
    avg = (float) sum / row_size;
    printf("Row %d, row average: %.2f\n", coords[0], avg);
  }

  /*Column sub-communicator creation */
  sum = 0;
  sub_coords[0] = 1;
  sub_coords[1] = 0;
  MPI_Cart_sub(cart_grid, sub_coords, &cart_col);
  MPI_Comm_size(cart_col, &col_size);
  MPI_Comm_rank(cart_col, &col_rank);

  /*Column sub-communicator's average calculation */
  MPI_Reduce(&world_rank, &sum, 1, MPI_INT, MPI_SUM, 0, cart_col);

  if (col_rank == 0) {
    avg = (float) sum / col_size;
    printf("Column %d, column average: %.2f\n", coords[1], avg);
  }

  /* Finalization operations */
  MPI_Comm_free(&cart_grid);
  MPI_Comm_free(&cart_col);
  MPI_Comm_free(&cart_row);
  MPI_Finalize();

  return 0;
}
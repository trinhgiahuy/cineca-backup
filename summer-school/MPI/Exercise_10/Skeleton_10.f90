program cartesian

use mpi

implicit none

  INTEGER :: world_rank, row_rank, col_rank, cart_rank  ! Why do we need all of these?
  INTEGER :: nprocs, row_size, col_size
  INTEGER :: dims(2), coords(2)
  LOGICAL :: period(2), sub_coords(2)
  INTEGER :: src_rank, dst_rank
  INTEGER :: sum, temp
  REAL :: avg

  ! Declare MPI variables: we need three new communicators that we will call cart_grid (main cartesian), cart_row (local row) and cart_col (local column). What is their type?
  ... 
  INTEGER :: status(MPI_STATUS_SIZE)
  INTEGER :: ierr

  !$ MPI Initialization
  CALL MPI_Init(ierr)
  CALL MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  CALL MPI_Comm_rank(MPI_COMM_WORLD, world_rank, ierr)

  !$ Cartesian grid creation. What do these initializations mean?
  dims(1) = 0
  dims(2) = 0
  period(1) = .true.
  period(2) = .true.

  ! Use MPI_Dims_create for the selection of the proper dimension array and create the cartesian communicator, to be stored in cart_grid
  CALL MPI_Dims_create(...)
  CALL MPI_Cart_create(...)

  !$ Local cart_rank initialization and comparison to global world_rank. Use MPI_Comm_rank as usual, but what to put instead of MPI_COMM_WORLD?
  CALL MPI_Comm_rank(...)

  WRITE (*,'(a,i1,a,i1,a)') 'I am world_rank ', world_rank, ' in MPI_COMM_WORLD and world_rank ', cart_rank , ' in the &
  cartesian communicator'

  !$ Coordinates creation and neighbour communication
  CALL MPI_Cart_coords(cart_grid, cart_rank, 2, coords, ierr)

  !$ Communication south. Complete MPI_Cart_shift so that is coherent with MPI_Sendrecv below. South= direction y, looking down
  sum = world_rank
  CALL MPI_Cart_shift(...)
  CALL MPI_Sendrecv(world_rank, 1, MPI_INTEGER, dst_rank, 0, temp, 1, MPI_INTEGER, src_rank, 0, cart_grid, status, ierr)
  sum = sum + temp

  !$ Communication north= direction y, looking up 
  CALL MPI_Cart_shift(...)
  CALL MPI_Sendrecv(world_rank, 1, MPI_INTEGER, dst_rank, 0, temp, 1, MPI_INTEGER, src_rank, 0, cart_grid, status, ierr)
  sum = sum + temp

  !$ Communication east= direction y, looking up 
  CALL MPI_Cart_shift(...)
  CALL MPI_Sendrecv(world_rank, 1, MPI_INTEGER, dst_rank, 0, temp, 1, MPI_INTEGER, src_rank, 0, cart_grid, status, ierr)
  sum = sum + temp

  !$ Communication west= direction x, looking left
  CALL MPI_Cart_shift(...)
  CALL MPI_Sendrecv(world_rank, 1, MPI_INTEGER, dst_rank, 0, temp, 1, MPI_INTEGER, src_rank, 0, cart_grid, status, ierr)
  sum = sum + temp

  !$ Neighbour's average
  avg = REAL(sum)/5
  WRITE (*,'(a,i2,a,i1,a,i1,a,f6.2)') 'Cart rank ', cart_rank, ' (', coords(1), ', ', coords(2), &
       '), neighbours average: ', avg

  !$ Row sub-communicator creation. How to fill sub_coords so that the correct dimension is dropped?
  sum = 0
  sub_coords(1) = ...
  sub_coords(2) = ...
  CALL MPI_Cart_sub(...) ! Decide what are the starting communicator and the new communicator. Use that info for the two commands below
  CALL MPI_Comm_size(..., row_size, ierr)
  CALL MPI_Comm_rank(..., row_rank, ierr)

  !$ Row sub-communicator's average calculation. First, the sum of all local ranks (use the variable "sum" for this)
  CALL MPI_Reduce(...)

  ! The sum calculated is divided for the number of elements in the row communicator
  if (row_rank.eq.0) then
    avg = REAL(sum) /row_size
    WRITE (*,'(a,i1,a,f6.2)') 'Row ',coords(1),' row average: ',avg
  endif

  !$ Column sub-communicator creation. How to fill sub_coords so that the correct dimension is dropped?
  sum = 0
  sub_coords(1) = ...
  sub_coords(2) = ...
  CALL MPI_Cart_sub(...) ! Decide what is the starting communicator and the new communicator. Use that info for the two commands below
  CALL MPI_Comm_size(..., col_size, ierr)
  CALL MPI_Comm_rank(..., col_rank, ierr)

  !$ Column sub-communicator's average calculation. First, the sum of all local ranks (use the variable "sum" for this)
  CALL MPI_Reduce(...)

  ! The sum calculated is divided for the number of elements in the column communicator
  if (col_rank.eq.0) then
    avg = REAL(sum) / col_size
    WRITE (*,'(a,i1,a,f6.2)') 'Column ',coords(2),' column average: ',avg
  endif

  !$ Finalization operations. Every communicator created must be freed: what is the command for this?
  CALL MPI_C...
  CALL MPI_C...
  CALL MPI_C...
  CALL MPI_Finalize(ierr)

end program cartesian

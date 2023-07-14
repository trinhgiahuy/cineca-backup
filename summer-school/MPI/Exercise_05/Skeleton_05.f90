program ghost_cells
    use mpi
    implicit none

    integer, parameter :: N=20
    integer :: my_rank, nprocs, ierr
    integer :: i,j
    integer :: rem, num_local_col
    integer :: proc_right, proc_left
    integer, allocatable :: matrix(:,:)
    integer status1(MPI_Status_size), status2(MPI_Status_size)

    ! Initialize the environment and store the size in "nprocs" and the rank in "my_rank"
    ...

    !  number of columns for each mpi task. How does it work? Try to write it down.
    rem= mod(N,nprocs)
    num_local_col = (N - rem)/nprocs

    if(my_rank < rem) num_local_col = num_local_col+1

    ! Allocation of the global matrix . Instead of "..." put one of the following:
    1. num_local_col  2. num_local_col+1  3. num_local_col+2

    allocate(matrix(N,...))

    ! inizialization of the local matrix
    matrix = my_rank

    ! Information about the neighbour processes (imagine that you are sending left and rught arrays of data, so the variables are named to reflect that)
    proc_right = my_rank+1
    proc_left = my_rank-1
    if(proc_right .eq. nprocs) proc_right = 0
    if(proc_left < 0) proc_left = nprocs-1

    ! check printings
    write(*,*) "my_rank, proc right, proc left ", my_id, proc_right, proc_left
    write(*,*) "my_rank, num_local_col ", my_rank, num_local_col
    write(*,*) "my_rank, matrix(1,1), matrix(1,num_local_col+2), matrix(N,num_local_col+2)", &
                my_rank, matrix(1,1), matrix(1,num_local_col+2), &
                matrix(N,num_local_col+2)

    ! send receive of the ghost regions
    ! First one: sends the leftiest real column to the proc on its left, and receives on the rightest ghost column from the proc on its right
    call mpi_sendrecv(...)

    ! Second one: sends the rightiest real column to the proc on its right, and receives on the leftiest ghost column from the proc on its left
    call mpi_sendrecv(...)

    ! check printings
    write(*,*) "my_rank ", my_rank, " colonna arrivata da sinistra: ", matrix(:,1)
    write(*,*) "my_rank ", my_rank, " colonna arrivata da destra: ", &
                matrix(:,num_local_col+2)

    deallocate(matrix)
    ! Finalize MPI environment
    call mpi_finalize(...)

end program
  

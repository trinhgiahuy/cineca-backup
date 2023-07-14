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

    call mpi_init(ierr)
    call mpi_comm_rank(MPI_COMM_WORLD,my_rank,ierr)
    call mpi_comm_size(MPI_COMM_WORLD,nprocs,ierr)

    !  number of columns for each mpi task

    rem= mod(N,nprocs)
    num_local_col = (N - rem)/nprocs

    if(my_rank < rem) num_local_col = num_local_col+1

    allocate(matrix(N,num_local_col+2))

    ! inizialization of the local matrix
    matrix = my_rank

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
    call mpi_sendrecv(matrix(:,2),N,MPI_INTEGER,proc_left,10, &
         matrix(:,(num_local_col+2)),N,MPI_INTEGER,proc_right,10, &
         MPI_COMM_WORLD,status1,ierr)

    call mpi_sendrecv(matrix(:,(num_local_col+1)),N,MPI_INTEGER,proc_right, &
         11,matrix(:,1),N,MPI_INTEGER,proc_left,11,MPI_COMM_WORLD,status2,ierr)

    ! check printings
    write(*,*) "my_rank ", my_rank, " colonna arrivata da sinistra: ", matrix(:,1)
    write(*,*) "my_rank ", my_rank, " colonna arrivata da destra: ", &
                matrix(:,num_local_col+2)

    deallocate(matrix)

    call mpi_finalize(ierr)

end program
  

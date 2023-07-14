program mpi_type_vector_test

! Declare variables
use mpi
implicit none
integer :: n_proc, n_rank, ierr, i,j 
integer, parameter :: n=5, nb=2
real :: a(n,n)
integer :: mystatus(MPI_STATUS_SIZE)

! We need to declare an MPI handle for the new datatype. What type will it be?
... :: myvector

! Start MPI
call MPI_Init(ierr)
call MPI_Comm_size(MPI_COMM_WORLD,n_proc,ierr)
call MPI_Comm_rank(MPI_COMM_WORLD,n_rank,ierr)

! Check the number of processes is 2
if(n_proc /= 2) then
  if(n_rank == 0) print*,'Test program has to work only with two MPI processes'
  call MPI_Finalize(ierr)
  STOP
endif

! Initialize matrix
if(n_rank == 0) a=0
if(n_rank == 1) a=1

! Define vector nd commit the new datatype in "myvector"
...
...

! Print matrix a for rank=1
if(n_rank == 1) then
  print*,'Matrix A before communications:'
  do i=1,n
  do j=1,n
     write(*,'(F10.2,1X)',advance='no') a(i,j)
  enddo
  write(*,*) 
  enddo
endif

! Communicate. Remember that we are sending one istance of the now datatype "myvector"
if(n_rank == 0) then
   call MPI_Send(...)
endif
if(n_rank == 1) then
   call MPI_Recv(...)
endif

! Print matrix a for rank=1
if(n_rank == 1) then
  print*,'Matrix A after communications:'
  do i=1,n
  do j=1,n
     write(*,'(F10.2,1X)',advance='no') a(i,j)
  enddo
  write(*,*) 
  enddo
endif

! Free the allocated vector datatype. How?
call MPI_Type_free(...)

! Finalize MPI
call MPI_Finalize(ierr)

end program mpi_type_vector_test

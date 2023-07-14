program mpi_type_vector_test

! Declare variables
use mpi
implicit none
integer :: n_proc, n_rank, ierr, i,j 
integer, parameter :: n=5, nb=2
real :: a(n,n)
integer :: mystatus(MPI_STATUS_SIZE)

integer :: myvector

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

! Define vector
call MPI_Type_vector(n,nb,n,MPI_REAL,myvector,ierr)
call MPI_Type_commit(myvector, ierr)

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

! Communicate
if(n_rank == 0) then
   call MPI_Send(a,1,myvector,1,100,MPI_COMM_WORLD, ierr)
endif
if(n_rank == 1) then
   call MPI_Recv(a,1,myvector,0,100,MPI_COMM_WORLD, mystatus, ierr)
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

call MPI_Type_free(myvector, ierr)

! Finalize MPI
call MPI_Finalize(ierr)

end program mpi_type_vector_test

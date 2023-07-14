program hello

  use mpi

  implicit none

  integer ierr,me,nprocs,left,right
  integer status(MPI_STATUS_SIZE)
  integer request

  integer,parameter :: ndata = 1000000

  real :: a(ndata)
  real :: b(ndata)

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)
  !$ Initialize data
  a = my_rank
  b = -1
  !$ Compute neighbour ranks. By using the mod operation, periodicity is guaranteed.
  right = mod(me + 1 , nprocs)
  left =  mod(me - 1 + nprocs , nprocs)
  !$ Sendrecv data

  call MPI_ISEND(a,ndata,MPI_REAL,right,0,MPI_COMM_WORLD,request,ierr)
  call MPI_RECV(b,ndata,MPI_REAL,left,0,MPI_COMM_WORLD,status,ierr)

  call MPI_WAIT(request,status,ierr)

  print *,'I am proc ',me,' and I have received b(1) = ',b(1)
  call MPI_FINALIZE(ierr)

end program hello

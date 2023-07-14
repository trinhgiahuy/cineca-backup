program hello

  use mpi

  implicit none

  integer ierr,me,nprocs,left,right
  integer status(MPI_STATUS_SIZE)

  integer,parameter :: ndata = 1000

  integer a, b, somma

  integer i

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)
  !$ Initialize workspace
  a = me
  b = -1
  !$ Compute neighbour ranks
  right = mod(me + 1 , nprocs)
  left  = mod(me - 1 + nprocs , nprocs)
  !$ Circular sum
  somma = a

  do i = 1,nprocs-1
     call MPI_SENDRECV(a,1,MPI_INT,left,0, &
          b,1,MPI_INT,right,0,MPI_COMM_WORLD,status,ierr)
     !$ Set "a" value to the newly received rank
     a = b
     !$ Update the partial sum
     somma = somma + a
  enddo

  print *,'I am proc ',me,' and somma = ',somma
  call MPI_FINALIZE(ierr)

end program hello

program hello

  !$ Use MPI module in Fortran 90
  use mpi
  implicit none

  integer ierr,me,nprocs

  !$ Initialize MPI environment
  call MPI_INIT(ierr)
  
  !$ Get the size of the MPI_COMM_WORLD communicator
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  
  !$ Get my rank... (zero-based even if we are in Fortran)
  call MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)
  
  !$ ...and print it.
  print *,"Hello, I am task ",me," of ",nprocs,"."
  
  !$  Finalize MPI environment
  call MPI_FINALIZE(ierr)

end program hello

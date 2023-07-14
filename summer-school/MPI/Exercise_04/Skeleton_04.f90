program hello

  !$ what to include to make MPI work?
  use ...

  implicit none

  integer ierr,me,nprocs,left,right
  integer status(MPI_STATUS_SIZE)

  integer,parameter :: ndata = 1000
  integer a, b, somma
  integer i

  !$ Initialize the MPI Environment and get the size and rank of each process.
  !$ The variable "nprocs" should contain the size, and the variable "me" should contain the rank
  call MPI_I...
  call MPI_C...
  call MPI_C...

  !$ Initialize workspace
  a = me   ! This will be the buffer sent to everyone
  b = -1   ! Temporary buffer to receive data from other tasks

  !$ Compute neighbour ranks. Try to figure out why this works for both numbers
  right = mod(me + 1 , nprocs)
  left  = mod(me - 1 + nprocs , nprocs)
  
  somma = a ! This will store the global sum updated at every loop. It is initialized with the rank of the process itself, that adds to the count

!$ Circular sum. Where should we stop the iteration? Complete the termination condition appropriately
  do i = 1,...
   !$ Insert the MPI communication function here. We want to send the content of the variable a to the process on our left, and to receive in our variable b the message from the process on our right. Can we do that with just one MPI routine?
     call MPI_S...
     !$ Set "a" value to the newly received rank, so it will the one that gets transferred in the next cycle
     a = b
     !$ Update the partial sum with the newly received rank (that is now also in "a")
     somma = somma + a
  enddo

  print *,'I am proc ',me,' and somma = ',somma
  !$ Finalize MPI environment
  call MPI_F...

end program hello

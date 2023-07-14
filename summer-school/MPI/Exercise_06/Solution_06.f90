program distribuite

  use mpi

  implicit none

  character(LEN=50) :: stringa

  integer, parameter :: N = 10

  integer ierr, error
  integer status(MPI_STATUS_SIZE)

  integer k, i, j, rem, iglob, Ncol, Nrow, sup
  integer me, nprocs

  integer, allocatable, dimension(:,:) :: a

  ! Initialize MPI environment and get size and rank in "nprocs" and "me" respectively
    ...

  !$ Number of columns that are assigned to each processor, taking care of the remainder
  Nrow = N
  Ncol = N / nprocs

  rem = MOD(N, nprocs)
  if (me < rem) then
     Ncol = Ncol + 1
  endif

  !$ Allocate local workspace (and notice that the array is distributed by columns)
  ALLOCATE( a(Nrow, Ncol) )
  !$ Column of the first "one" entry. It sets up a starting position from which every rank should check the elements of their local matrix
  iglob = (Ncol * me) + 1
  if (me >= rem) then
     iglob = iglob + rem
  endif

  !$ Initialize local matrix
  do j=1,Ncol
     do i=1,Nrow
        if (i == iglob) then
           A(i,j) = 1.0
        else
           A(i,j) = 0.0
        endif
     enddo
     iglob = iglob + 1;  ! Why?
  enddo

  write(stringa, *) Nrow
  !$ Print matrix
  ! Set up the "if" condition so that rank 0 receives from everyone and prints the final matrix, and the other ranks are sending their local matrix to it.
  if (me == 0) then
     !$ Rank 0: print local buffer 
     do j=1,Ncol
        print '('//trim(stringa)//'(I2))', A(:,j)
     enddo

     !$ Receive new data from other processes
     !$ in an ordered fashion and print the buffer
     do k=1, nprocs-1
        if (k==rem) then
           Ncol = Ncol - 1
        endif
        ! This is the receiver task. Set up the receive call
        ...
        do j=1,Ncol
           print '('//trim(stringa)//'(I2))', A(:,j)
        enddo
     enddo
  else
     !$ Send local data to Rank 0
     ...
  endif
  DEALLOCATE(a)
  ! Finalize the environment
  call MPI_FINALIZE(ierr)

end program distribuite

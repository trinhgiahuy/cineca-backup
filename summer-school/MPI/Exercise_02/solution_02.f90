program hello

  use mpi

  implicit none

  integer ierr, me, nprocs, ierr
  integer status(MPI_STATUS_SIZE)

  integer,parameter :: ndata = 10000
  real              :: a(ndata)
  real              :: b(ndata)

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)

  !$ Initialize data 
  a = me
  !$ Protect against the use with a number of processes != 2
  if (nprocs .ne. 2) then
     if (me==0) then
        print *,"This program must run on 2 processors"
     endif
     call MPI_FINALIZE(ierr)
     stop
  endif
  
  !$ Send and Receive data    
  if (me==0) then
     call MPI_SEND(a,ndata,MPI_REAL,1,0,MPI_COMM_WORLD,ierr)
     call MPI_RECV(b,ndata,MPI_REAL,1,0,MPI_COMM_WORLD,status,ierr)
  else
     call MPI_RECV(b,ndata,MPI_REAL,0,0,MPI_COMM_WORLD,status,ierr)
     call MPI_SEND(a,ndata,MPI_REAL,0,0,MPI_COMM_WORLD,ierr)
  endif

  print *,'I am proc ',me,' and I have received b(1) = ',b(1)
  call MPI_FINALIZE(ierr)

end program hello

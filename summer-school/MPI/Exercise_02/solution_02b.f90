program hello
  implicit none
  include 'mpif.h'

  integer ierr,me,nprocs,you,req
  integer status(MPI_STATUS_SIZE)

  integer,parameter :: ndata = 10000
  real :: a(ndata)
  real :: b(ndata)

  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, me, ierr)

  a = me

  if (nprocs .ne. 2) then
     if (me==0) then
        print *,"This program must run on 2 processors"
     endif
     call MPI_FINALIZE(ierr)
     stop
  endif

  !If me=0 then you=1; if me=1 then you=0
  you = 1-me

  call MPI_ISEND(a,ndata,MPI_REAL,you,0,MPI_COMM_WORLD,req,ierr)
  call MPI_RECV(b,ndata,MPI_REAL,you,0,MPI_COMM_WORLD,status,ierr)

  call MPI_WAIT(req,status,ierr)

  print *,'I am task ',my_rank,' and I have received b(1) = ',b(1)
  call MPI_FINALIZE(ierr)

end program hello

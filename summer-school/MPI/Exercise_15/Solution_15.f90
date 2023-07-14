subroutine trans (a, n)
!* transpose square matrix a, dimension nxn
!* Consider this as a black box for the MPI course
!*/
  integer i, j, ij, ji, l,n
  double precision tmp
  double precision a(*)

  ij = 0
  l = -1
  do i = 0, n-1
      l = l+ n + 1
      ji = l
      ij = ij+i + 1
      do j=i+1,n-1 
	  tmp = a(ij+1)
	  a(ij+1) = a(ji+1)
	  a(ji+1) = tmp
	  ij = ij+1
	  ji = ji+ n
      enddo	
  enddo 
  return
end subroutine trans
!* 
!* This program demonstrates the use of MPI_Alltoall when
!* transposing a square matrix.
!* For simplicity, the number of processes is 4 and the dimension
!* of the matrix is fixed to NROW
!*/
 program main
  implicit none
  include 'mpif.h'
  integer :: nprocs
  integer :: proc_me,proc_up,proc_down
    integer ierr
    integer,parameter::NROW=131072
    integer,parameter::NP=1024
    integer,parameter::NBLK=NROW/NP
    double precision,parameter::ONEML=99999999999.0
    double precision,allocatable:: a(:,:)
    double precision,allocatable:: b(:,:)

  integer i, j, rank,provided,itemp
  double precision r0
  call MPI_INIT(ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD,nprocs,ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD,proc_me,ierr)
  allocate( a(NBLK,NROW),b(NBLK,NROW))

  if(proc_me.eq.0) write(*,*) 'Transposing a',NROW,'x',NROW,' matrix, divided among',NP,'processors'
  if (nprocs.ne.NP) then
      if (proc_me.eq.0) then
	write(*,*) 'Error, number of processes must be',NP
      end if
      call MPI_Finalize (ierr)
	stop
   endif
 r0 = MPI_Wtime()
  do j=1,NROW
     do i=1,NBLK
      a(i,j) = ONEML * j + i + NBLK *proc_me 
     enddo
  enddo
 r0 = MPI_Wtime()-r0
 if(proc_me.eq.0) write(*,*) 'Building matrix time (sec)',r0

  !* do the MPI part of the transpose */
  !/* Tricky here is the number of items to send and receive. 
  ! * Not NROWxNBLK as one may guess, but the amount to send to one process
  ! * and the amount to receive from any process */

 r0 = MPI_Wtime()
  !/* MPI_Alltoall does not a transpose of the data received, we have to
  ! * do this ourself: */
  call MPI_Alltoall (a(1,1), NBLK * NBLK, MPI_DOUBLE_PRECISION, b(1,1), NBLK * NBLK, MPI_DOUBLE_PRECISION, MPI_COMM_WORLD,ierr)

 r0 = MPI_Wtime()-r0
  if(proc_me.eq.0) write(*,*) 'MPI_Alltoall time (sec)',r0

  !* transpose NP square matrices, order NBLKxNBLK: */
r0 = MPI_Wtime()
  do i=1,NP
     call trans(b(1,(i-1) * NBLK+1),NBLK)
  enddo
r0 = MPI_Wtime()-r0;
  if(proc_me.eq.0) write(*,*) 'Transpose block matrices time (sec)',r0

   !* now check the result */
  do j=1,NROW  
     do i=1,NBLK
	if (b(i,j).ne. ONEML * (i + NBLK * proc_me) + j ) then
	    write(*,*) 'process',proc_me,'b',b(i,j),'expected', ONEML * (i + NBLK * proc_me) + j
	    call MPI_Abort (MPI_COMM_WORLD,1, ierr)
	endif 
     enddo
  enddo
  if (proc_me.eq.0) write(*,*) 'Transpose seems ok!'
  call MPI_Finalize (ierr)
end program main
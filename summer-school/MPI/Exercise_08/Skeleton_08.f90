PROGRAM main
 IMPLICIT NONE
 include 'mpif.h'

 INTEGER, parameter::N=50
 INTEGER ::my_id, num_procs, ierr
 INTEGER ::i, num_elem ! count of elements that will be used later for the scatter/gather functions
 INTEGER, dimension(N)::array, array_final
 INTEGER, allocatable, dimension(:)::array_recv
 INTEGER, allocatable:: sendcount(:), displs(:)

 CALL MPI_INIT( ierr ) ! starts MPI
 CALL MPI_COMM_RANK( MPI_COMM_WORLD, my_id, ierr ) ! get current process id
 CALL MPI_Comm_size ( MPI_COMM_WORLD, num_procs, ierr ) ! get number of processes

 ! proc 0 initializes the principal array
 IF( my_id .eq. 0) THEN
   DO i=1,N
     array(i)= i
   END DO
 END IF

! Calculate the number of elements to distribute; it depends on the number of processes involved
 num_elem= N/num_procs
 IF (my_id < MOD(N,num_procs)) THEN
    num_elem = num_elem +1
 ENDIF

 ALLOCATE(array_recv(num_elem))

 ! in case that N is a multiple of the number of MPI tasks, the same number of
 ! elements is send to (and received from) by root process to others, so
 ! mpi_scatter and mpi_gather are called
IF ( MOD(N,num_procs) .eq. 0 ) THEN
 ! Insert MPI_Scatter here. What to scatter and how many elements per block?
    CALL MPI_SCATTER(...)

    WRITE(*,*) "my_id", my_id, "elementi ricevuti:", array_recv(1:num_elem)

    DO i=1,num_elem
      array_recv(i)= array_recv(i)+my_id ! Update of the received array as requested by the exercise
    END DO

 ! Insert MPI_Gather here. What to gather and how many elements per block?
    CALL MPI_GATHER(...)

    IF( my_id .eq. 0) THEN
       WRITE(*,*) "N is multiple of num_procs, mod(N,num_procs)= ", &
mod(N,num_procs)
       WRITE(*,*) " array finale: ", array_final(:)
    END IF

ELSE

!in case that N is not a multiple of the number of MPI tasks,
!mpi_scatterv and mpi_gatherv have to be used

    ALLOCATE(sendcount(num_procs),displs(num_procs))

! Calculation of the arrays of sendcounts and displacements. How do they work?
    displs(1) = 0
    sendcount=N/num_procs
    if(0<mod(N,num_procs)) sendcount(1)=N/num_procs+1

    DO i=2,num_procs
      if( (i-1) < mod(N,num_procs) ) sendcount(i) = N/num_procs +1
      displs(i) = SUM(sendcount(1:i-1))
    END DO

IF (my_id .eq. 0 ) THEN
       WRITE(*,*) "sendcount: ", sendcount
       WRITE(*,*) "displs: ", displs
    END IF

! Insert MPI_Scatterv here. Remember what parameters does it need? What is the difference with MPI_Scatter?
    CALL MPI_SCATTERV(...)

    WRITE(*,*) "my_id", my_id, "elementi ricevuti:", array_recv(1:num_elem)

    DO i=1,num_elem
      array_recv(i)= array_recv(i)+my_id
    END DO

! Insert MPI_Gatherv here. Remember what parameters does it need? What is the difference with MPI_Gather?
    CALL MPI_GATHERV(...)

    IF( my_id .eq. 0) THEN
       WRITE(*,*) "N is not multiple of num_procs, mod(N,num_procs)= ", &
mod(N,num_procs)
       WRITE(*,*) " array finale: ", array_final(:)
    END IF
ENDIF

DEALLOCATE(array_recv)
CALL MPI_FINALIZE(ierr)
END PROGRAM main

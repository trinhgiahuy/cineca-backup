program main 

use mpi
implicit none
 
integer rank, size, to, from, tag, count, i, ierr 
integer src, dest 
integer st_source, st_tag, st_count 
integer status(MPI_STATUS_SIZE) 
double precision data(100) 

 
call MPI_INIT( ierr ) 
call MPI_COMM_RANK( MPI_COMM_WORLD, rank, ierr ) 
call MPI_COMM_SIZE( MPI_COMM_WORLD, size, ierr ) 
 print *, 'Process ', rank, ' of ', size, ' is alive' 
 dest = size - 1 
     src = 0 
!       
      if (rank .eq. src) then 
         to     = dest 
         count  = 10 
         tag    = 2001 
         do  i=1, 10 
            data(i) = i 
         enddo
         call MPI_SEND( data, count, MPI_DOUBLE_PRECISION, to, &
                         tag, MPI_COMM_WORLD, ierr ) 
      else if (rank .eq. dest) then 
         tag   = MPI_ANY_TAG 
         count = 10   
         from  = MPI_ANY_SOURCE 
         call MPI_RECV(data, count, MPI_DOUBLE_PRECISION, from, & 
                        tag, MPI_COMM_WORLD, status, ierr )  

      endif

call MPI_Finalize(ierr)
end program  main

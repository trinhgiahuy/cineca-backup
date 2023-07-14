! 
! my version of mpi_send
!
subroutine MPI_Send( start, count, datatype, dest, tag, comm, ierr ) 
 integer start(*), count, datatype, dest, tag, comm 
 write(*,*) 'MPI send called '
 call PMPI_send( start, count, datatype, dest, tag, comm, ierr ) 
end 


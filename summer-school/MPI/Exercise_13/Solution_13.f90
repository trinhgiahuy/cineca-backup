PROGRAM mpi2io

     USE MPI
     IMPLICIT none

     INTEGER :: myrank, nproc,ierr;
     INTEGER, PARAMETER :: dim_buf = 10
     INTEGER :: buf(dim_buf)
     INTEGER :: i, intsize;
     INTEGER(KIND=MPI_OFFSET_KIND) :: offset, file_size
     INTEGER :: fh
     INTEGER :: status(MPI_STATUS_SIZE)
     INTEGER :: filetype

     CALL MPI_Init(ierr);
     CALL MPI_Comm_size(MPI_COMM_WORLD,nproc,ierr)
     CALL MPI_Comm_rank(MPI_COMM_WORLD,myrank,ierr)

     DO i=1,dim_buf
        buf(i) = myrank*dim_buf+i-1;
     END DO

     ! Open the file and write by using individual file pointers 
     CALL MPI_File_open(MPI_COMM_WORLD, "output.dat", MPI_MODE_CREATE+MPI_MODE_WRONLY, MPI_INFO_NULL,fh,ierr)

     CALL MPI_Type_size(MPI_INTEGER,intsize,ierr)
     offset = myrank*dim_buf*(intsize)
     CALL MPI_File_seek(fh,offset,MPI_SEEK_SET,ierr)
     CALL MPI_File_write(fh,buf,dim_buf,MPI_INTEGER,status,ierr)

     CALL MPI_File_close(fh,ierr)

     ! Re-open the file and read by using explicit offset
     CALL MPI_File_open(MPI_COMM_WORLD,"output.dat",MPI_MODE_RDONLY,MPI_INFO_NULL,fh,ierr)

     CALL MPI_File_get_size(fh,file_size,ierr)
     offset = file_size/nproc*myrank
     write(6,*) "myid ",myrank,"filesize ", file_size, "offset ", offset

     CALL MPI_File_read_at(fh,offset,buf,dim_buf,MPI_INTEGER,status,ierr)
     CALL MPI_File_close(fh,ierr)

     ! Write the new file using the mpi_type_create_vector. Use the fileview 
     CALL MPI_File_open(MPI_COMM_WORLD, "output_mod.dat", MPI_MODE_CREATE+MPI_MODE_WRONLY, MPI_INFO_NULL,fh,ierr)
     CALL MPI_Type_vector(dim_buf/2,2,2*nproc,MPI_INTEGER,filetype,ierr)
     CALL MPI_Type_commit(filetype,ierr)

     CALL MPI_Type_size(MPI_INTEGER, intsize, ierr)
     offset = 2*intsize*myrank
     CALL MPI_File_set_view(fh,offset,MPI_INTEGER,filetype,"native",MPI_INFO_NULL,ierr)
     CALL MPI_File_write_all(fh,buf,DIM_BUF,MPI_INTEGER,status,ierr)

     CALL MPI_File_get_size(fh,file_size,ierr)
     write(6,*) "myid ", myrank, "filesize of the second file written", file_size, "offset", offset

     CALL MPI_Type_free(filetype,ierr)
     CALL MPI_File_close(fh,ierr) 
     CALL MPI_Finalize(ierr)

END PROGRAM mpi2io


#include <stdlib.h>
#include <stdio.h>

#include "mpi.h"

#define PARALLEL_IO_PATH_TO_FILE "./output_c.dat"
#define ind(i,j) (i*lsizes[1]+j)


/* Warning for C programmers: depending on the compiler of your choice, you may need to add the flag -std=c99 because of some variable definitions that
   require C99 standard. */

int main( int argc, char *argv[] )
{
//  Declare variables (or do it later)
    const int m = 10; // rows of global matrix
    const int n = 10; // cols of global matrix

//  Start MPI
    MPI_Init( &argc, &argv );

//  Set cartesian topology
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dims[2]; dims[0] = 0; dims[1] = 0;
    MPI_Dims_create(world_size, 2, dims);

    int periods[2]; periods[0] = 0; periods[1] = 0;

    MPI_Comm     comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm);

    int rank, coords[2];
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 2, coords);
    if (rank == 0) {
      printf("\nUsing a grid of [%d][%d] processes\n",
          dims[0], dims[1]);
    }

    // set subarray info
    int rem;
    int gsizes[2], psizes[2];
    int lsizes[2], start_indices[2];

    gsizes[0] = m;  /* no. of rows in global array */
    gsizes[1] = n;  /* no. of columns in global array*/

    psizes[0] = dims[0];  /* no. of processes in vertical dimension  of process grid */
    psizes[1] = dims[1];  /* no. of processes in horizontal dimension  of process grid */

    lsizes[0] = m/psizes[0];   /* no. of rows in local array */
    rem = m%psizes[0];
    if (rem && coords[0] < rem) {
      lsizes[0]++;
      start_indices[0] = coords[0] * lsizes[0];
    } else {
      start_indices[0] = rem + coords[0] * lsizes[0];
    }

    lsizes[1] = n/psizes[1];   /* no. of columns in local array */
    rem = n%psizes[1];
    if (rem && coords[1] < rem) {
      lsizes[1]++;
      start_indices[1] = coords[1] * lsizes[1];
    } else {
      start_indices[1] = rem + coords[1] * lsizes[1];
    }

    // initialize local matrix (array) 
    int local_array_size = lsizes[0] * lsizes[1];
    int *local_array = (int *) malloc( local_array_size * sizeof(int) );
    for (int i=0; i<lsizes[0]; i++) 
        for (int j=0; j<lsizes[1]; j++) 
            local_array[ind(i,j)] = n*(i+start_indices[0]+1)+(j+start_indices[1]+1);

    /* create subarray filetype for MPI File view */
    MPI_File     fh;
    MPI_Datatype filetype;
    MPI_Status   f_status;

//  Create subarray, open and write file in parallel
    MPI_Type_create_subarray(2, gsizes, lsizes, start_indices,
                         MPI_ORDER_C, MPI_INT, &filetype);
    MPI_Type_commit(&filetype);

    MPI_File_delete(PARALLEL_IO_PATH_TO_FILE, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, PARALLEL_IO_PATH_TO_FILE,
      MPI_MODE_CREATE | MPI_MODE_WRONLY,  MPI_INFO_NULL, &fh);
    MPI_Offset displ = 0;
    MPI_File_set_view(fh, displ, MPI_INT, filetype, "native",  MPI_INFO_NULL);

    MPI_File_write_all(fh, local_array, local_array_size,
           MPI_INT, &f_status);

    MPI_File_close(&fh);

    // Parallel verify 
    int errcount = 0;
    int *verify_array = (int *) calloc( local_array_size , sizeof(int) );

    MPI_File_open(MPI_COMM_WORLD, PARALLEL_IO_PATH_TO_FILE,
      MPI_MODE_RDONLY,  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, displ, MPI_INT, filetype, "native",  MPI_INFO_NULL);
    MPI_File_read_all(fh, verify_array, local_array_size,
           MPI_INT, &f_status);
    for (int i=0; i<lsizes[0]; i++) 
        for (int j=0; j<lsizes[1]; j++) 
            if (verify_array[(i,j)] != local_array[(i,j)]) {
               fprintf(stderr,"Parallel Verify ERROR: at indexes %d,%d read=%d, written=%d \n",
                       i, j, verify_array[ind(i,j)], local_array[ind(i,j)]);
               errcount++;
            }
    if ( errcount == 0 ) {
      printf("Parallel Verify: test passed on proc(%d)\n", rank);
    }

    MPI_File_close(&fh);

    MPI_Type_free(&filetype);
    free(local_array);
    free(verify_array);

    // Serial verify 
    errcount = 0;
    if(rank == 0) {
       FILE* fh;
       fh = fopen(PARALLEL_IO_PATH_TO_FILE,"rb");
       int *serial_verify_array = (int *) calloc(n , sizeof(int) );
       for (int i=0; i<m; i++) {
           fread(serial_verify_array,sizeof(int),n,fh);
           for (int j=0; j<n; j++) {
               if (serial_verify_array[j] != n*(i+1)+(j+1)) {
                   fprintf(stderr,"Serial Verify ERROR: at index %d,%d read=%d, written=%d \n",
                          i, j, serial_verify_array[j], n*(i+1)+(j+1));
                  errcount++;
               }
           }
       }

       if (!errcount) {
         printf("Serial Verify: test passed\n");
       }

       free(serial_verify_array);
       fclose(fh);
    }


//  MPI Finalize
    MPI_Finalize();

    return 0;
}
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

int main (int argc, char** argv) {
  int me, nprocs, microbuff ;

  MPI_Init(&argc, &argv) ;

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs) ;
  MPI_Comm_rank(MPI_COMM_WORLD, &me) ;

  microbuff=2000000 ;
  if(me==0){
    printf("Elaboration %s running on %3d cores...\n", argv[1], nprocs) ;
    usleep(microbuff) ;
  }
  MPI_Barrier(MPI_COMM_WORLD) ;

  MPI_Finalize() ;
  return 0;
}

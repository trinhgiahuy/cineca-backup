#include <stdlib.h>
#include <stdio.h>

/*what to include to make MPI work?*/
#include <...>

int main(int argc, char* argv[]){

    int me, nprocs, left, right, count;
    MPI_Status status;

    int a;
    int b;
    int sum;

    /* Initialize the MPI Environment and get the size and rank of each process.
       The variable "nprocs" should contain the size, and the variable "me" should contain the rank */
    MPI_I...
    MPI_C...
    MPI_C...

    /* Initialize workspace */
    a   = me;  //This will be the buffer sent to everyone
    b   = -1; // Temporary buffer to receive data from other tasks
    sum = a;  //This will store the global sum updated at every loop. It is initialized with the rank of the process itself, that adds to the count

    /* Compute neighbour ranks. Try to figure out why this works for both numbers */
    right = (me+1)%nprocs;
    left  = (me-1+nprocs)%nprocs;

    /* Circular sum. Where should be stop the iteration? Complete the termination condition appropriately */
    for(count = 1; count < ...; count++)
      {
    /* Insert the MPI communication function here. We want to send the content of the variable a to the process on our left, and to receive in our variable b the message from the process on our right. Can we do that with just one MPI function?
    NOTE: Remember also how to exchange simple variables instead of arrays */
    MPI_S...
    /* Set "a" value to the newly received rank, so it will the one that gets transferred in the next cycle */
    a    = b;
    /* Update the partial sum with the newly received rank (that is now also in "a") */
    sum += a;
      }
    printf("\tI am proc %d and sum(0) = %1.2f \n", me, sum);

   /* Finalize the MPI environment */
    MPI_F...
    return 0;
}

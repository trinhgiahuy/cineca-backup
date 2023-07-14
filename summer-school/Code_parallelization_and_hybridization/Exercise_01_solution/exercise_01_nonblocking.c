/*
 *    It evolves the equation:
 *                            u,t + u,x + u,y = 0
 *    Using a Lax scheme.
 *    The initial data is a cruddy gaussian.
 *    Boundaries are flat: copying the value of the neighbour
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NX 100
#define NY 100
#define LX 2.0
#define LY 2.0

#define sigma 0.1
#define tmax 100



/* conversions from discrete to real coordinates */
float ix2x(int ix){
    return ((ix - 1) - (NX-1) / 2.0)*LX/(NX-1); 
}


/*
 * THE FOLLOWING CHANGES: every processor has a different offset.
 * I also pass proc_y as an argument instead of proc_me since it will
 * become useful when saving the output files. 
 */

float iy2y(int iy, int proc_y, int nprocs){ 
    return ((iy-1) - (NY-1) / 2.0 + proc_y*(NY / nprocs) + (proc_y < NY % nprocs? proc_y:NY % nprocs)) * LY/(NY-1); 
}

/* Function for evaluating the results that calculates the sum of the array values */
float summa(int nx, int ny, float* val){
float summma=0.0;
int ix,iy;
    for(iy=1;iy<=ny;++iy)
        for(ix=1;ix<=nx;++ix){
          summma+=val[((nx+2)*iy)+ix];
 }
  return(summma);
}



/* 
 * initialize the system with a gaussian temperature distribution 
 */

int init_transport(float *temp, int NLY, int proc_me,int nprocs){
    
    int ix,iy;
    float x,y;

    for(iy=1;iy<=NLY;++iy)
	for(ix=1;ix<=NX;++ix){	
	    x=ix2x(ix);
	    y=iy2y(iy, proc_me, nprocs);
	    temp[((NX+2)*iy)+ix] = tmax*exp((-((x*x)+(y*y)))/(2.0*(sigma*sigma)));
	}
    return 0;
}

/*
 * save the temperature distribution
 * the ascii format is suitable for splot gnuplot function
 */

int save_gnuplot(char *filename, float *temp, int NLY, int proc_me, int nprocs){

    float *buf;
    int ix, iy, iproc=0;
    FILE *fp;

    MPI_Status status;

    if(proc_me == 0){

	buf = (float *) malloc ((NX+2)*(NLY+2)*sizeof(float));
	fp = fopen(filename, "w");

	for(iy=1;iy<=NLY;++iy){		
	    for(ix=1;ix<=NX;++ix){
		fprintf(fp, "\t%f\t%f\t%g\n", ix2x(ix), iy2y(iy, proc_me, nprocs), temp[((NX+2)*iy)+ix]);
	    }
	    fprintf(fp, "\n");
	}

        for(iproc=1; iproc<nprocs; iproc++){
          int nly, count;
	  MPI_Recv(buf, (NX+2)*(NLY+2), MPI_REAL, iproc, 0, MPI_COMM_WORLD, &status);
	  MPI_Get_count( &status, MPI_REAL, &count );
	  nly = count / (NX + 2) - 2;
 
          for(iy=1;iy<=nly;++iy){
            for(ix=1;ix<=NX;++ix){
              fprintf(fp, "\t%f\t%f\t%g\n", ix2x(ix), iy2y(iy, iproc, nprocs), buf[((NX+2)*iy)+ix]);
            }
 
            fprintf(fp, "\n");
          }
        }
	fclose(fp);
        free(buf);
    }

    else{
	MPI_Send(temp, (NX+2)*(NLY+2), MPI_REAL, 0, 0, MPI_COMM_WORLD);
    }

    return 0;
}

int update_boundaries_FLAT(float *temp, int NLY, int nprocs, int proc_me, int proc_up, int proc_down){

    MPI_Status status[4];
    MPI_Request request[4];
    int iy=0, ix=0;
    
    for(iy=1;iy<=NLY;++iy){
	temp[(NX+2)*iy] = temp[((NX+2)*iy)+1];
	temp[((NX+2)*iy)+(NX+1)] = temp[((NX+2)*iy)+NX];
    }

/*  
 * only the lowest has the lower boundary condition  
 */
    if (proc_me==0) for(ix=0;ix<=NX+1;++ix) temp[ix] = temp[(NX+2)+ix];
    
/*
 * only the highest has the upper boundary condition  
 */
    if (proc_me==nprocs-1) for(ix=0;ix<=NX+1;++ix) temp[((NX+2)*(NLY+1))+ix] = temp[((NX+2)*(NLY))+ix];


/*  
 *communicate the ghost-cells  
 *
 *  
 * lower-down  
 */
 
 MPI_Isend(&temp[(NX+2)+1], NX, MPI_REAL, proc_down, 0,MPI_COMM_WORLD,&request[0]);
 MPI_Irecv(&temp[((NX+2)*(NLY+1))+1], NX, MPI_REAL, proc_up, MPI_ANY_TAG, MPI_COMM_WORLD, &request[1]);

/*
 * higher-up  
 */
  MPI_Isend(&temp[((NX+2)*(NLY))+1], NX, MPI_REAL, proc_up, 0,MPI_COMM_WORLD,&request[2]);
  MPI_Irecv(&temp[1], NX, MPI_REAL, proc_down, MPI_ANY_TAG, MPI_COMM_WORLD,&request[3]);


 MPI_Waitall(4,request,status);

    return 0;
}


int evolve(float dtfact, float *temp, float *temp_new, int NLY){
    
    float dx, dt;
    int ix, iy;
    float temp0;
    
    dx = 2*LX/NX;
    dt = dtfact*dx/sqrt(3.0);
    
    for(iy=1;iy<=NLY;++iy)
	for(ix=1;ix<=NX;++ix){
	    temp0 = temp[((NX+2)*iy)+ix];
	    temp_new[((NX+2)*iy)+ix] = temp0-0.5*dt*(temp[((NX+2)*(iy+1))+ix]-temp[((NX+2)*(iy-1))+ix]+
            temp[((NX+2)*iy)+(ix+1)]-temp[((NX+2)*iy)+(ix-1)])/dx;
	}
    
    for(iy=0;iy<NLY+2;++iy)
	for(ix=0;ix<NX+2;++ix)
	    temp[((NX+2)*iy)+ix] = temp_new[((NX+2)*iy)+ix];
    
    return 0;
}


int main(int argc, char* argv[]){

    int nprocs, proc_me, proc_up, proc_down;

    int i=0, NLY;
    float *temp, *temp_new,before,after;


    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_me);
    
    /* Check if  nprocs is compatible with NY */
    if ( NY < nprocs ) {
      if (proc_me==0) {
	printf("\nTrying to run with %d processes, while the maximum allowed size is %d (NY)\n",nprocs,NY);
      }
      MPI_Finalize();
    }
/*  
 *  all the communications from/to MPI_PROC_NULL do nothing
 */

    proc_down = proc_me-1;
    proc_up = proc_me+1;

    if(proc_down < 0)
	proc_down = MPI_PROC_NULL;
    
    if(proc_up == nprocs) proc_up = MPI_PROC_NULL;

    NLY = NY/nprocs;
    if (proc_me < NY % nprocs)
      NLY++;
    
    temp = (float *) malloc ((NX+2)*(NLY+2)*sizeof(float));
    temp_new = (float *) malloc ((NX+2)*(NLY+2)*sizeof(float));

    init_transport(temp, NLY, proc_me,nprocs);
    update_boundaries_FLAT(temp, NLY, nprocs, proc_me, proc_up, proc_down);

    float tbefore =summa(NX, NLY, temp);
    MPI_Reduce(&tbefore, &before, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(proc_me==0) {
	   printf(" sum temp before: %f\n",before);
    }

    save_gnuplot("transport.dat", temp, NLY, proc_me, nprocs);   

    for(i=1;i<=500;++i){
	evolve(0.1, temp, temp_new, NLY);
	update_boundaries_FLAT(temp, NLY, nprocs, proc_me, proc_up, proc_down);
    }
    
    save_gnuplot("transport_end.dat", temp, NLY, proc_me, nprocs);   

        float tafter =summa(NX, NLY, temp);
	MPI_Reduce(&tafter, &after, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(proc_me==0) {
	   printf(" sum temp after: %f\n",after);
    }

    free(temp);
    free(temp_new);
    MPI_Finalize();

/*    return 0; */
}
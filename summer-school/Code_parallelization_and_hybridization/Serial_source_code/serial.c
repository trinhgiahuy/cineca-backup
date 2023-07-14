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

#define NX 100
#define NY 100
#define LX 2.0
#define LY 2.0

#define sigma 0.1
#define tmax 100


/* 
 * conversions from discrete to real coordinates
 */
float ix2x(int ix){
    return ((ix-1) - (NX-1) / 2.0)*LX/(NX-1);
}

float iy2y(int iy){ 
    return ((iy-1) - (NY-1) / 2.0)*LY/(NY-1);
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
int init_transport(float *temp){ 
    int ix,iy;
    float x,y;

    for(iy=1;iy<=NY;++iy)
	for(ix=1;ix<=NX;++ix){
	    x=ix2x(ix);
	    y=iy2y(iy);
	    temp[((NX+2)*iy)+ix] = tmax*exp((-((x*x)+(y*y)))/(2.0*(sigma*sigma)));
	}
    return 0;
}

/*
 * save the temperature distribution
 * the ascii format is suitable for splot gnuplot function
 */
int save_gnuplot(char *filename, float *temp) {
    int ix,iy;
    FILE *fp;

    fp = fopen(filename, "w");   
    for(iy=1;iy<=NY;++iy){		
	for(ix=1;ix<=NX;++ix)
	    fprintf(fp, "\t%f\t%f\t%g\n", ix2x(ix), iy2y(iy), temp[((NX+2)*iy)+ix]);
	fprintf(fp, "\n");
    }
    fclose(fp);

    return 0;
}

int update_boundaries_FLAT(float *temp){
    int ix=0, iy=0;

    for(iy=0;iy<=NY+1;++iy){
	temp[(NX+2)*iy] = temp[((NX+2)*iy)+1];
	temp[((NX+2)*iy)+(NX+1)] = temp[((NX+2)*iy)+NX];
    }

    for(ix=0;ix<=NX+1;++ix){
	temp[ix] = temp[(NX+2)+ix];
	temp[((NX+2)*(NY+1))+ix] = temp[((NX+2)*NY)+ix];
    }
    return 0;
}


int evolve(float dtfact, float *temp, float *temp_new){
    float dx, dt;
    int ix, iy;
    float temp0;

    dx = 2*LX/NX;
    dt = dtfact*dx/sqrt(3.0);
    for(iy=1;iy<=NY;++iy)
	for(ix=1;ix<=NX;++ix){
	    temp0 = temp[((NX+2)*iy)+ix];
	    temp_new[((NX+2)*iy)+ix] = temp0-0.5*dt*(temp[((NX+2)*(iy+1))+ix]
            -temp[((NX+2)*(iy-1))+ix]+temp[((NX+2)*iy)+(ix+1)]-temp[((NX+2)*iy)+(ix-1)])/dx;
	}

    for(iy=0;iy<=NY+1;++iy)
	for(ix=0;ix<=NX+1;++ix)
	    temp[((NX+2)*iy)+ix] = temp_new[((NX+2)*iy)+ix];


    return 0;
}

int main(int argc, char* argv[]){
    int i=0, nRow=NX+2, nCol=NY+2;
    float *temp, *temp_new;


    temp = (float *) malloc (nRow*nCol*sizeof(float));
    temp_new = (float *) malloc (nRow*nCol*sizeof(float));

    init_transport(temp);
    update_boundaries_FLAT(temp);

    float before=summa(NX, NY, temp);
    printf(" sum temp before: %f\n",before);

    save_gnuplot("transport.dat", temp);

    for(i=1;i<=500;++i) {
        evolve(0.1, temp, temp_new);
        update_boundaries_FLAT(temp);
    }

    float after=summa(NX, NY, temp);
    printf(" sum temp after: %f\n",after);
    save_gnuplot("transport_end.dat", temp);


    free(temp);
    free(temp_new);    
    return 0;
}
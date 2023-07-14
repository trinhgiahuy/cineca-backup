#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Structre of Arrays

#define N 1024

void main() {

        int i;
        float sum=0.0;
        float x,y,z;
     
	struct node {
	   float x[N];
	   float y[N];
	   float z[N];
	};

	struct node NODES;

        for (i=0;i<N;i++) {
	   NODES.x[i]=1;
	   NODES.y[i]=1;
	   NODES.z[i]=1;
	 }
        for (i=0; i<N; i++) {
	   x=NODES.x[i];
	   y=NODES.y[i];
	   z=NODES.z[i];
	   sum+=sqrtf(x*x+y*y+z*z);
	}

}



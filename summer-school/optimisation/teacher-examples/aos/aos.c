#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// Array of structures

#define N 1024

void main() {

        int i;
        float sum=0.0;
        float x,y,z;
     
	struct node {
	   float x,y,z;
	   int n;
	};

	struct node NODES[N];

        for (i=0;i<N;i++) {
	   NODES[i].x=1;
	   NODES[i].y=1;
	   NODES[i].z=1;
	 }
        for (i=0; i<N; i++) {
	   x=NODES[i].x;
	   y=NODES[i].y;
	   z=NODES[i].z;
	   sum+=sqrtf(x*x+y*y+z*z);
	}

}



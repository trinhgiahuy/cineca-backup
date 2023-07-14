#include <stddef.h>
#include <stdlib.h>
#include <math.h>

void add_arrays(float* a, float* b, size_t n)
{
    size_t i;
    /* WORK ON THIS PART */
    float agg = 0.0;
    float max = (float) n;
    
    for (i = 0; i < n; ++i)
    {
        a[i] += exp(b[i]);
        agg += b[i];
        if (agg > max)
            break;
    }
    /* WORK ON THE ABOVE PART */
}

void main() {
   float *a, *b;
   size_t i;
   size_t n=500;
   
   a = malloc(n*sizeof(float));
   b = malloc(n*sizeof(float));

   for (i = 0; i < n; ++i) {
   a[i] = 0.;
   b[i] = 1.;
   }

   add_arrays(a, b, n);

   free(a); free(b);

}


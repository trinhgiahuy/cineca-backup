# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <fftw3.h>

int main ( void )

{
  ptrdiff_t i;
  const ptrdiff_t n = 1024;
  fftw_complex *in;
  fftw_complex *out;
  fftw_complex *newout;
  fftw_plan plan_backward;
  fftw_plan plan_forward;
/* Create arrays. */
   in = fftw_malloc ( sizeof ( fftw_complex ) * n );
   out = fftw_malloc ( sizeof ( fftw_complex ) * n );
   newout = fftw_malloc ( sizeof ( fftw_complex ) * n );
/* Initialize data */
   for ( i = 0; i < n; i++ )
   { 
      if (i >= (n/4-1) && (3*n/4-1)) 
        {
          in[i][0] = 1.;
          in[i][1] = 0.;
        } 
      else
        {
          in[i][0] = 0.;
          in[i][1] = 0.;
        }       
   }
/* Create plans. */
   plan_forward = fftw_plan_dft_1d ( n, in, out, FFTW_FORWARD, FFTW_ESTIMATE );
   plan_backward = fftw_plan_dft_1d ( n, out, newout, FFTW_BACKWARD, FFTW_ESTIMATE );
/* Compute transform (as many times as desired) */
   fftw_execute ( plan_forward );
/* Normalization */
   for ( i = 0; i < n; i++ )
   { 
   in[i][0] = in[i][0]/n;
   in[i][1] = in[i][1]/n;
   }
/* Compute anti-transform */
   fftw_execute ( plan_backward );
/* Print results */
   for ( i = 0; i < n; i++ )
   { 
    printf("i = %d, in = (%f), out = (%f,%f), newout = (%f)\n",i,in[i],out[i][0],out[i][1],newout[i]);
   }
/* deallocate and destroy plans */
   fftw_destroy_plan ( plan_forward );
   fftw_destroy_plan ( plan_backward );
   fftw_free ( in );
   fftw_free ( newout );
   fftw_free ( out );

  return 0;
}



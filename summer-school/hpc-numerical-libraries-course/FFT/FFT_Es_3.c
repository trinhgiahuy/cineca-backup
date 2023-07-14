# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <mpi.h>
# include <fftw3-mpi.h>
     
     int main(int argc, char **argv)
     {
         const ptrdiff_t L = 128, M = 128, N = 128;
         fftw_plan plan;
         fftw_complex *data ;
         ptrdiff_t alloc_local, local_L, local_L_start, i, j, k,  i;
         double xx, yy, zz, rr, r2, t0, t1, t2, t3, tplan, texec;
         const double amp = 0.25;
         /* Initialize */
         MPI_Init(&argc, &argv);
         fftw_mpi_init();
     
         /* get local data size and allocate */
         alloc_local = fftw_mpi_local_size_3d(L, M, N, MPI_COMM_WORLD, &local_L, &local_L_start);
         data = fftw_alloc_complex(alloc_local);
         /* create plan for in-place forward DFT */
         t0 = MPI_Wtime();
         plan = fftw_mpi_plan_dft_2d(L, M, N, data, data, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
         t1 = MPI_Wtime();
         /* initialize data to some function my_function(x,y) */
         for (i = 0; i < local_L; ++i)for (j = 0; j < M; ++j) for (k = 0; k < N; ++k)
         {
	    ii = i + local_L_start;
            xx = ( (double) ii - (double)L/2 )/(double)L;
            yy = ( (double)j - (double)M/2 )/(double)M;
            zz = ( (double)k - (double)N/2 )/(double)M;
            r2 = pow(xx, 2) + pow(yy, 2) + pow(zz,2);
            rr = sqrt(r2);
            if (rr <= amp)
            {
              data[(i*M*N) + (j*N) + k][0] = 1.;   
              data[((i*M*N) + (j*N) + k) + j][1] = 0.; 
            }
            else
            {
              data[(i*M*N) + (j*N) + k][0] = 0.;   
              data[(i*M*N) + (j*N) + k][1] = 1.; 
            }
         }
         /* compute transforms, in-place, as many times as desired */
         t2 = MPI_Wtime();
         fftw_execute(plan);
         t3 = MPI_Wtime();
         /* Print results */
         tplan = t1 - t0;
         texec = t2 - t1;
         printf(" T_plan = %f, T_exec = %f \n",tplan,texec);
         /* deallocate and destroy plans */ 
         fftw_destroy_plan(plan);
         fftw_mpi_cleanup();
         fftw_free ( data );
         MPI_Finalize();
     }

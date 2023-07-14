      program FFTW_MPI_3D
        use, intrinsic :: iso_c_binding
        implicit none
	include 'mpif.h'
        include 'fftw3-mpi.f03'
        integer(C_INTPTR_T), parameter :: L = 128
        integer(C_INTPTR_T), parameter :: M = 128
        integer(C_INTPTR_T), parameter :: N = 128
        type(C_PTR) :: plan, cdata
        complex(C_DOUBLE_COMPLEX), pointer :: fdata(:,:,:)
        integer(C_INTPTR_T) :: alloc_local, local_N, local_k_offset
        integer(C_INTPTR_T) :: i, j, k
        complex(C_DOUBLE_COMPLEX) :: fout
        integer :: ierr, myid, nproc, ncorestot
        real(C_DOUBLE) :: t1, t2, t3, t4, tplan, texec
! Initialize
	call mpi_init(ierr)
	call MPI_COMM_SIZE(MPI_COMM_WORLD, nproc, ierr)
	call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
	call fftw_mpi_init()
	ncorestot = nproc

!   get local data size and allocate (note dimension reversal)
	  alloc_local = fftw_mpi_local_size_3d(N, M, L, &
     &                MPI_COMM_WORLD, local_N, local_k_offset)
	  cdata = fftw_alloc_complex(alloc_local)
	  call c_f_pointer(cdata, fdata, [L, M, local_N])
!   create MPI plan for in-place forward DFT (note dimension reversal)
          t1 = MPI_wtime()
	  plan = fftw_mpi_plan_dft_3d(N, M, L, fdata, fdata, &
     &         MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE)
          t2 = MPI_wtime()
! initialize data to some function my_function(i,j)
          do k = 1, local_N
	    do j = 1, M
	      do i = 1, L
	        call initial(i, j, (k + local_k_offset), L, M, N, fout)
	        fdata(i, j, k) = fout
	      end do
            end do
	  end do
! compute transform (as many times as desired)
          t3 = MPI_wtime()
	  call fftw_mpi_execute_dft(plan, fdata, fdata)
          t4 = MPI_wtime()
! print solutinos
          tplan = t2-t1
          texec =t4-t3
          print*,'Tplan=',tplan,'   Texec=',texec
! deallocate and destroy plans
	  call fftw_destroy_plan(plan)
	  call fftw_mpi_cleanup()
	  call fftw_free(cdata)
	call mpi_finalize(ierr)
!
      end
!
! ***** Subroutines *****************************************
!
      subroutine initial(i, j, k, L, M, N, fout)
      use, intrinsic :: iso_c_binding
      integer(C_INTPTR_T), intent(in) :: i, j, k, L, M, N
      complex(C_DOUBLE_COMPLEX), intent(out) :: fout
      real(C_DOUBLE), parameter :: amp = 0.125
      real(C_DOUBLE) :: xx, yy, zz, LL, MM, NN, r1

        xx = real(i, C_DOUBLE) - real((L+1)/2, C_DOUBLE)
	yy = real(j, C_DOUBLE) - real((M+1)/2, C_DOUBLE)
	zz = real(k, C_DOUBLE) - real((N+1)/2, C_DOUBLE)
	LL = real(L, C_DOUBLE)
	MM = real(M, C_DOUBLE)
        NN = real(N, C_DOUBLE)

        r1 = sqrt(((xx/LL)**2.) + ((yy/MM)**2.) + ((yy/MM)**2.))
	if (r1 .le. amp) then
	  fout = CMPLX(1., 0. , C_DOUBLE_COMPLEX)
        else
          fout = CMPLX(0., 0. , C_DOUBLE_COMPLEX)
        endif

      return
      end


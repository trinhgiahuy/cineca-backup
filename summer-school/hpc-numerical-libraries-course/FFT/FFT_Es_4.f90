       PROGRAM FFT_3D_2Decomp_MPI
       use mpi
       use, intrinsic :: iso_c_binding
       use decomp_2d
       use decomp_2d_fft
      implicit none
      integer, parameter :: L = 128
      integer, parameter :: M = 128
      integer, parameter :: N = 128
      integer, parameter :: p_row = 16
      integer, parameter :: p_col = 16
      integer :: nx, ny, nz
      complex(mytype), allocatable, dimension(:,:,:) :: in, out
      complex(mytype) :: fout
      integer :: ierror, i,j,k, numproc, mype
      integer, dimension(3) :: sizex, sizez

! ===== Initialize
       call MPI_INIT(ierror)
       call MPI_Comm_size(MPI_COMM_WORLD, numproc, ierror) 
       call MPI_COMM_RANK(MPI_COMM_WORLD, mype, ierror) 

       call decomp_2d_init(L,M,N,p_row,p_col)
       call decomp_2d_fft_init

       do i =1, 3
          sizex(i) = xend(i) - xstart(i) + 1
          sizez(i) = zend(i) - zstart(i) + 1
       end do

       allocate (in(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)))
       allocate (out(zstart(1):zend(1),zstart(2):zend(2),zstart(3):zend(3)))

! ===== each processor gets its local portion of global data =====

       do k=xstart(3),xend(3)
          do j=xstart(2),xend(2)
             do i=xstart(1),xend(1)
                call initial(i, j, k, L, M, N, fout)
                in(i,j,k) = fout
             end do
          end do
        end do

! ===== 3D forward FFT =====
        call decomp_2d_fft_3d(in, out, DECOMP_2D_FFT_FORWARD)
! ==========================

        call decomp_2d_fft_finalize
        call decomp_2d_finalize
        deallocate(in,out)
        call MPI_FINALIZE(ierror)

      end
!
! ***** Subroutines *****************************************
!
      subroutine initial(i, j, k, L, M, N, fout)
      use, intrinsic :: iso_c_binding
      use decomp_2d
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

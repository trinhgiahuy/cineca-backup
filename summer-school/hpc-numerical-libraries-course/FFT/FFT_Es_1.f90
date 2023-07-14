      program FFTW1D
        use, intrinsic :: iso_c_binding
        implicit none
        include 'fftw3.f03'
        integer(C_INTPTR_T):: L = 1024 
        integer(C_INT) :: LL
        type(C_PTR) :: plan1, plan2
        type(C_PTR) :: p_idata, p_odata, p_newdata
        complex(C_DOUBLE_COMPLEX), dimension(:), pointer :: odata
        complex(C_DOUBLE), dimension(:), pointer :: idata, newdata
        integer :: i
        character(len=41), parameter :: filename='serial_data_optim.txt'
!! Allocate
           LL = int(L,C_INT)
           p_idata = fftw_alloc_complex(L)
           p_odata = fftw_alloc_complex(L)
           p_newdata = fftw_alloc_complex(L)
           call c_f_pointer(p_idata,idata,(/L/))
           call c_f_pointer(p_odata,odata,(/L/))
           call c_f_pointer(p_newdata,newdata,(/L/))
!!   create MPI plan for in-place forward DF
           plan1 = fftw_plan_dft_1d(LL, idata, odata, FFTW_FORWARD, FFTW_ESTIMATE)
           plan2 = fftw_plan_dft_1d(LL, odata, newdata, FFTW_BACKWARD,FFTW_ESTIMATE)
!! initialize data 
          do i = 1, L 
              if ( (i .ge. (L/4)) .and. (i .le. (3*L/4)) ) then
                  idata(i) = (1.,0.)
              else
                  idata(i) = (0.,0.)
              endif
          end do
!! compute transform (as many times as desired)
          call fftw_execute_dft(plan1, idata, odata)
!! Normalizzation
          odata = odata/L
!! Compute anti-transform
          call fftw_execute_dft(plan2, odata, newdata)
!! print data
          OPEN(UNIT=45, FILE=filename, ACCESS='SEQUENTIAL')
              do i = 1, L
                   write(45,*) i, idata(i),odata(i),newdata(i)
              end do
          CLOSE(45)
!! deallocate and destroy plans
          call fftw_destroy_plan(plan1)
          call fftw_destroy_plan(plan2)
          call fftw_free(p_idata)
          call fftw_free(p_odata)
          call fftw_free(p_newdata)
      end

!
!     It evolves the equation:
!                             u,t + u,x + u,y = 0
!     Using a Lax scheme.
!     The initial data is a cruddy gaussian.
!     Boundaries are flat: copying the value of the neighbour
!

module transport
  implicit none
  save

  integer, parameter :: NX=100
  integer, parameter :: NY=100
  real, parameter    :: LX=2.0
  real, parameter    :: LY=2.0

  real temp(0:NX+1 , 0:NY+1)
  real temp_new(0:NX+1 , 0:NY+1)


CONTAINS
  !conversions from discrete to real coordinates

  real function ix2x(ix)
    integer ix
    ix2x = ((ix-1)-(NX-1)/ 2.0)*LX/(NX-1)
  end function ix2x

  real function iy2y(iy)
    integer iy
    iy2y = ((iy-1)-(NY-1)/ 2.0)*LY/(NY-1)
  end function iy2y


 ! initialize the system with a gaussian temperature distribution

  subroutine init_transport
    integer ix,iy
    real x,y
    real,parameter :: sigma = 0.1
    real,parameter :: tmax = 100

    do iy=1,NY
       do ix=1,NX
          x=ix2x(ix)
          y=iy2y(iy)
          temp(ix,iy) = tmax*exp(-(x**2+y**2)/(2.0*sigma**2))
       enddo
    enddo

  end subroutine init_transport

  ! save the temperature distribution
  ! the ascii format is suitable for splot gnuplot function
  subroutine save_gnuplot(filename)

    character(len=*) filename
    integer ix,iy

    open(unit=20,file=filename,form='formatted')
    do iy=1,NY
       do ix=1,NX
          write(20,*) ix2x(ix),iy2y(iy),temp(ix,iy)
       enddo
       write(20,*)
    enddo

    close(20)
  end subroutine save_gnuplot

  subroutine update_boundaries_FLAT
    temp(0   , 1:NY) = temp(1  , 1:NY)
    temp(NX+1, 1:NY) = temp(NX   , 1:NY)

    temp(1:NX , 0)    = temp(1:NX , 1)
    temp(1:NX , NY+1) = temp(1:NX , NY)
  end subroutine update_boundaries_FLAT

  subroutine evolve(dtfact)
    real :: dtfact
    real :: dx,dt
    integer :: ix,iy
    real :: temp0


    dx = 2*LX/NX
    dt = dtfact*dx/sqrt(3.0)
    do iy=1,NY
       do ix=1,NX
          temp0 = temp(ix,iy)
          temp_new(ix,iy) = temp0 - 0.5 * dt * (temp(ix+1,iy)-temp(ix-1,iy)+temp(ix,iy+1)-temp(ix,iy-1)) / dx
       enddo
    enddo

    temp = temp_new

  end subroutine evolve
end module transport

program prova
  use transport
  implicit none

  integer i

  call init_transport
  call update_boundaries_FLAT
  write(*,*) 'sum temp before',sum(temp(1:NX,1:NY))
  call save_gnuplot('transport.dat')

  do i=1,500
     call evolve(0.1)
     call update_boundaries_FLAT
  enddo

  call save_gnuplot('transport_end.dat')
  write(*,*) 'sum temp after',sum(temp(1:NX,1:NY))

end program prova

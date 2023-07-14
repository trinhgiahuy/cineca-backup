!
!     It evolves the equation:
!                             u,t + u,x + u,y = 0
!     Using a Lax scheme.
!     The initial data is a cruddy gaussian.
!     Boundaries are flat: copying the value of the neighbour
!
! parallel WITHOUT TOPOLOGIES: distributed by columns
! dynamic allocation to decide run-time the size to allocate
! no reminder for teaching purposes

module comms
  implicit none
  save
  include 'mpif.h'

  integer :: nprocs
  integer :: proc_me,proc_up,proc_down

CONTAINS
  subroutine INIT_COMMS
    integer ierr

    call MPI_COMM_SIZE(MPI_COMM_WORLD,nprocs,ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD,proc_me,ierr)

    proc_up = proc_me + 1
    proc_down = proc_me - 1

    ! all the communications from/to MPI_PROC_NULL do nothing
    if (proc_down < 0) proc_down=MPI_PROC_NULL
    if (proc_up >= nprocs) proc_up=MPI_PROC_NULL

  end subroutine INIT_COMMS
end module comms

module transport

  use comms
  implicit none
  save

  integer, parameter :: NX=100
  integer, parameter :: NY=100
  real, parameter    :: LX=2.0
  real, parameter    :: LY=2.0
  integer            :: NLY

  real,allocatable   ::  temp(:,:)
  real,allocatable   ::  temp_new(:,:)

CONTAINS
  !conversions from discrete to real coordinates
  real function ix2x(ix)
    integer ix
    ix2x = ((ix-1)-(NX-1)/2.0)*LX/(NX-1)
  end function ix2x

 ! THE FOLLOWING CHANGES: every processor has a different offset.
 ! I also pass proc_y as an argument instead of proc_me since it will
 ! become useful when saving the output files.

  real function iy2y(iy,proc_y)

    integer proc_y
    integer iy

    if (proc_y < mod(NY,nprocs)) then
      iy2y = (iy-1 - (NY-1)/2.0 + proc_y*(NY/nprocs) + proc_y) * LY/(NY-1)
    else
      iy2y = (iy-1 - (NY-1)/2.0 + proc_y*(NY/nprocs) + mod(NY,nprocs)) * LY/(NY-1)
    endif
  end function iy2y

  ! initialize the system with a gaussian temperature distribution
  ! in parallel, allocate the system too
   subroutine init_transport
    use comms

    integer ix,iy
    real x,y
    real,parameter :: sigma = 0.1
    real,parameter :: tmax = 100

    NLY = NY/nprocs
    if(proc_me < mod(NY,nprocs)) NLY=NLY+1

    allocate (     temp(0:NX+1 , 0:NLY+1) )
    allocate ( temp_new(0:NX+1 , 0:NLY+1) )

    !! DO loops on local indeces only
    do iy=1,NLY
       do ix=1,NX
          x=ix2x(ix)
          y=iy2y(iy,proc_me)
          temp(ix,iy) = tmax*exp(-(x**2+y**2)/(2.0*sigma**2))
       enddo
    enddo

  end subroutine init_transport

  ! save the temperature distribution
  ! the ascii format is suitable for splot gnuplot function
  ! collective, use temp_new as temporary buffer

  subroutine save_gnuplot(filename)
    character(len=*) filename
    integer ix,iy
    integer nly,count_el
    integer iproc
    integer ierr
    integer status(MPI_STATUS_SIZE)

    if (proc_me == 0) then
       open(unit=20,file=filename,form='formatted')

       do iy=1,NLY
          do ix=1,NX
             write(20,*) ix2x(ix),iy2y(iy,0),temp(ix,iy)
          enddo
          write(20,*)
       enddo

       do iproc=1,nprocs-1
          call MPI_RECV(temp_new,(NX+2)*(NLY+2),MPI_REAL,iproc,0, &
               MPI_COMM_WORLD,status,ierr)
          call MPI_GET_COUNT(status,MPI_REAL,count_el,ierr)
          nly = count_el / (NX + 2) - 2
          do iy=1,nly
             do ix=1,NX
                write(20,*) ix2x(ix),iy2y(iy,iproc),temp_new(ix,iy)
             enddo
             write(20,*)
          enddo
       enddo
       close(20)

    else
       call MPI_SEND(temp,(NX+2)*(NLY+2),MPI_REAL,0,0,MPI_COMM_WORLD,ierr)
    endif
  end subroutine save_gnuplot

  !! NY => NLY
  subroutine update_boundaries_FLAT
    integer status(MPI_STATUS_SIZE)
    integer ierr

    temp(0   ,  1:NLY) = temp(1  , 1:NLY)
    temp(NX+1, 1:NLY) = temp(NX   , 1:NLY)

    !! only the lowest has the lower boundary condition
    if (proc_me==0)        temp(1:NX , 0)    = temp(1:NX , 1)
    !! only the highest has the upper boundary condition
    if (proc_me==nprocs-1) temp(1:NX , NLY+1) = temp(1:NX , NLY)

    !! communicate the ghost-cells
    !! lower-down
    call MPI_SENDRECV(temp(1,1),NX,MPI_REAL,proc_down,0, &
         temp(1,NLY+1),NX,MPI_REAL,proc_up,0,MPI_COMM_WORLD,status,ierr)

    !! higher-up
    call MPI_SENDRECV(temp(1,NLY),NX,MPI_REAL,proc_up,0, &
         temp(1,0),NX,MPI_REAL,proc_down,0,MPI_COMM_WORLD,status,ierr)

  end subroutine update_boundaries_FLAT

  subroutine evolve(dtfact)
    real dtfact
    real dx,dt
    integer ix,iy
    real temp0

    dx = 2*LX/NX
    dt = dtfact*dx/sqrt(3.0)
    do iy=1,NLY
       do ix=1,NX
          temp0 = temp(ix,iy)
          temp_new(ix,iy) = temp0 - 0.5 * dt * &
               (temp(ix+1,iy)-temp(ix-1,iy)+temp(ix,iy+1)-temp(ix,iy-1)) / dx
       enddo
    enddo

    temp = temp_new

  end subroutine evolve
end module transport

program prova
  use transport
  use comms
  implicit none

  integer i
  integer ierr
  real before,tbefore,after,tafter

  call MPI_INIT(ierr)
  call init_comms
  call init_transport
  call update_boundaries_FLAT

  tbefore=sum(temp(1:NX,1:NLY))
  call MPI_Reduce(tbefore,before,1,MPI_REAL,MPI_SUM,0,MPI_COMM_WORLD,ierr)
  if (proc_me==0)  write(*,*) 'sum temp before',before

  call save_gnuplot('transport.dat')
  do i=1,500
     call evolve(0.1)
     call update_boundaries_FLAT
  enddo
  call save_gnuplot('transport_end.dat')

  tafter=sum(temp(1:NX,1:NLY))
  call MPI_Reduce(tafter,after,1,MPI_REAL,MPI_SUM,0,MPI_COMM_WORLD,ierr)
  if (proc_me==0)  write(*,*) 'sum temp after',after

  call MPI_FINALIZE(ierr)

end program prova
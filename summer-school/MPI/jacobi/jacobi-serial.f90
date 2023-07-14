program jacobi

  implicit none

  real(8), parameter :: top=1.0,bottom=10.0, left=1.0,right=1.0
  integer, parameter :: max_iter=1000000
  integer, parameter :: nprint=100
  integer :: nx,ny
  integer :: i,j,k,iter

  real(8) :: norm,bnorm
  real(8) :: tol=1e-4
  real(8), allocatable, dimension(:,:) :: grid,grid_new 
  character(len=32) :: arg


  if (command_argument_count() /= 2) then
     print *, &
          "You must provide two command line arguments, the global size in X and the global size in Y"
     stop
  end if

  call get_command_argument(1, arg)
  read(arg,*) nx
  call get_command_argument(2, arg)
  read(arg,*) ny

  print *,'grid size ',nx, ' x ',ny
  allocate(grid(0:ny+1,0:nx+1),grid_new(0:ny+1,0:nx+1) )


  ! boundary conditions
  grid(0,:)=left
  grid(ny+1,:)=right
  grid(:,0) = top
  grid(:,nx+1)=bottom

  grid_new=grid

  ! starting conditions
  grid(1:ny,1:nx)=0.0

  !! initial norm value

  norm=0.0
  do j=1,nx
     do i=1,ny
        norm=norm+((grid(i,j)*4-grid(i-1,j)-grid(i+1,j)-grid(i,j-1)-grid(i,j+1))**2)
     enddo
  enddo

  bnorm=sqrt(norm)
  !print *,'bnorm=',bnorm

  do iter=1, max_iter

     norm=0.0
     do j=1,nx
        do i=1,ny
           norm=norm+((grid(i,j)*4-grid(i-1,j)-grid(i+1,j)-grid(i,j-1)-grid(i,j+1))**2)
        enddo
     enddo

     norm=sqrt(norm)/bnorm

     if (norm .lt. tol) exit

     do j=1,nx
        do i=1,ny
           grid_new(i,j)=0.25 * (grid(i-1,j) + grid(i+1,j) + grid(i,j-1) + grid(i,j+1))
        end do
     end do

     grid=grid_new

     if (mod(iter,nprint)==0) then
        write(*,*) 'Iteration ',iter,' Relative norm ',norm
     endif

  enddo
  write(*,*) 'Terminated on ',iter, ' iterations, relative norm=',norm

  deallocate(grid,grid_new)

end program jacobi


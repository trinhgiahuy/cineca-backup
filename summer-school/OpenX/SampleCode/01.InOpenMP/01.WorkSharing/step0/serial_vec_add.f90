
program vecAdd
 
  implicit none
  real(kind=8) :: start, finish

  integer :: N=1e8
  real(kind=8), allocatable :: A(:), B(:), C(:)
  integer :: i
  
  ! Allocate memory
  allocate( A(N), B(N), C(N))

  ! Initilise data
  do i = 1, N
    A(i) = i*1.0_8
    B(i) = A(i)
  end do

  ! Start timer
  call cpu_time(start)

  do i = 1, N 
     C(i) = A(i) + B(i)
  end do

  call cpu_time(finish)

  ! Print result
  write(*,"(A)")         "------------------------------------"
  write(*,"(A,F10.3)")   "runtime:  ", finish-start
  write(*,"(A)")         "------------------------------------"

  deallocate(A,B, C)
end program vecAdd


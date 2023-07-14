program loop_order
  implicit none
  integer :: i,j,k
  integer, parameter :: n=1000

  real :: a(n,n), b(n,n), c(n,n)

  a=1.0
  b=2.0
  c=0.0

  do i=1,n
     do k=1,n
        do j=1,n
           c(i,j) = c(i,j) + a(i,k)*b(k,j)
        end do
     end do
  end do


end program loop_order

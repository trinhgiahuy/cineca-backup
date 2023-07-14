subroutine mysum(a,b,sum)

integer :: i
integer, parameter :: nmax=10000
real :: a(nmax),b(nmax),sum

sum=0.0

do i=1,nmax

   sum = sum + a(i)*b(i)

enddo

end subroutine mysum


subroutine vec

integer, parameter ::n=1000
integer :: i
real :: a(n),b(n),c(n)

do i=2,n
   a(i)=a(i-1)+1
enddo

end subroutine


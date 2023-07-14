subroutine test(a,b,n)

implicit none
integer :: n,i,j
real :: a(n),b(n)

integer :: c(n)

do i=1,n
   c(i)=i
enddo

do j=1,n

    a(c(j)) = a(j) + b(j)
enddo

end subroutine test

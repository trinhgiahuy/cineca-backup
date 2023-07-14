subroutine mult (a,b,c,n)

   integer :: i,n
   real :: a(n),b(n),c(n)

   do i=1,n
        a(i)=b(i)*c(i)
   enddo

end subroutine

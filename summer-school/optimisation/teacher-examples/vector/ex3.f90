subroutine add(a,n,j)
   integer :: i,j,n
   real :: a(n)

   do i=2,n
      a(i)=a(i-j)+1
   enddo

end subroutine add


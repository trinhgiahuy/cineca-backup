subroutine add(a,n)
   integer :: i,n
   real :: a(n)

   do i=2,n
      a(i-1)=a(i)+1
   enddo

end subroutine add


! this will not vectorise because j could be negative
! without using the simd directive make it vectorize 

subroutine change(a,j,k,n)

   real :: a(n)
   integer :: i,j,k,n

   do i=5,n
      a(i) = a(i+j)*k
   enddo


end subroutine

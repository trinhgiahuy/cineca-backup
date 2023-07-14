subroutine mult(a,b,n)
implicit none
integer :: n,i,j,k
real :: a(n,n),b(n,n),c(n,n)

  do i=1,N
     do k=1,N
       do j=1,N
        c(i,j) = c(i,j) + a(i,k)*b(k,j)
       enddo
     enddo
  enddo

end subroutine mult

program matrix

   integer, parameter :: N=3
   real(8) :: a(N,N),b(N,N),c(N,N)
   integer :: i,j,k

  do i=1,N
    do j=1,N
      a(i,j) = i
      b(i,j) = j
      c(i,j)=0.d0
    enddo
  enddo

  do i=1,N
     do k=1,N
       do j=1,N
        c(i,j) = c(i,j) + a(i,k)*b(k,j)
       enddo
     enddo
  enddo



end program matrix




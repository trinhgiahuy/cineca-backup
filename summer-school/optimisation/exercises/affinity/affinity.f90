program affinity
integer, parameter :: n=1000000000
real(4), allocatable :: a(:),b(:),c(:)
real(8), external :: mysecond
real(8) :: t0,t1

integer :: i,j
allocate(a(n),b(n),c(n))
!$omp parallel do
      do i=1,n
         a(i)=10.0
         b(i)=2.0
         c(i)=1.0
      enddo
!$omp end parallel do
      t0=mysecond()

   do j=1,10
!$OMP parallel do
      do i=1,n
         a(i)=b(i)+d*c(i)
      enddo
!$OMP end parallel do
   enddo
      t1=mysecond()
   write(*,*) 'Time taken ',t1-t0

 deallocate(a,b,c)

end program affinity

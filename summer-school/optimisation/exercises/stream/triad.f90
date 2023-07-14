program triad

implicit none
integer, parameter :: N=10

integer :: i, j,k
double precision, dimension(N):: a,b,c,d
real(8) :: start_time, stop_time,flops
integer, parameter :: niter=1000
real(8), external :: mysecond
integer :: ops

k=n*n
a=1.d0
b=a
c=a
d=a

start_time=mysecond()

do j=1, niter

   do i=1, n

      a(i) = b(i) +c(i) *d(i)

   enddo
   if (k .eq. 0) call dummy(a,b,c,d)
enddo

stop_time=mysecond()

!dops=16  for Haswell 16 DP flops/cycle
ops =2 ! 2 ops in the loop (non-vector)
flops=real(ops*n*niter)/(stop_time-start_time)

write(*,*) 'time ',stop_time-start_time
write(*,*) 'Mflops ',flops/1.0d6

end program triad
subroutine dummy(a,b,c,d)

   double precision :: a(:),b(:),c(:),d(:)
   a=b+c+d

end subroutine dummy


program saxpy
 
 implicit none
 integer :: i, N 
 real(kind=8), allocatable :: x(:), y(:)
 real :: a,start_time, end_time
 a = 2.0d0
 N = 1000
 
 allocate(x(n),y(n))

! Parallelize this block of code (optional) 
 do i = 1, N
  x = 1.0d0
  y = 2.0d0
 end do 
 
 call cpu_time(start_time)
 do i = 1, N
  y(i) = y(i) + a * x(i) 
 end do
 call cpu_time(end_time)
 deallocate(x,y)

 print '(a,f8.6)', 'SAXPY Time: ', end_time - start_time

end program saxpy



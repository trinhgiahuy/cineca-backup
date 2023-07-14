      program vecadd

!...  Prepared by NERSC User Services Group
!...  April, 2014

      implicit none
!      integer :: n = 5184
      integer :: n
      integer :: itmax = 10000000
#ifdef REAL4
      real, allocatable :: a(:), b(:), c(:)
#else
      real*8, allocatable :: a(:), b(:), c(:)
#endif
      integer i, it
      integer*8 c1, c2, cr, cm
      real*8 dt

      read(*,*) n
      allocate (a(n), b(n), c(n))

!...  Initialization

      do i=1,n
         a(i) = cos(i * 0.1)
         b(i) = sin(i * 0.1)
         c(i) = 0.
      end do

!...  Main loop

      call system_clock(c1, cr, cm)
      do it=1,itmax
         do i=1,n
            c(i) = a(i) + b(i)
         end do
         b(n) = sin(b(n))
      end do
      call system_clock(c2, cr, cm)

      dt = dble(c2-c1)/dble(cr)
      !print *, c(1)+c(n/2)+c(n), dt
      print *,  dt

      deallocate(a, b, c)
      end

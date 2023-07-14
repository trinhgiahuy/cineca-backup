program fibonacci

        implicit none

        integer :: x, y
        x = 10
        y = fib(x)

        print*, y

        contains

        RECURSIVE INTEGER FUNCTION fib(n) RESULT(res)
                INTEGER n, i, j
                IF ( n .LT. 2) THEN
                        res = n
                ELSE
              
                 i = fib( n-1 )
              

               
                 j = fib( n-2 )
              

              
                res = i+j
              END IF
        end function


end program fibonacci
PROGRAM datatype

USE omp_lib

implicit none

double precision, allocatable,dimension(:,:,:,:,:) :: A,B,C
!integer(kind=1), allocatable,dimension(:,:,:) :: D

integer :: t,i,j,k,l,m,size = 50,repetition=40
double precision :: time,time1

ALLOCATE(A(size,size,2,size,2), B(2,size,2,size,size), C(2,size,size,size,2))


do i=1,size
	do j=1,size
		do k=1,size
			do l=1,size
				do m=1,size
				A(m,l,MOD(k,1)+1,j,MOD(i,1)+1) = DBLE(k)/DBLE(i)
				B(MOD(m,1)+1,l,MOD(k,1)+1,j,MOD(i,1)+1) = DBLE(k)/DBLE(j)
				enddo
			enddo
		enddo
	enddo
enddo


time = omp_get_wtime()
do t = 1,repetition
do i=1,size-1
	do j=2,size-1
		do k=2,size-1
			do l=2,size-1
				do m=2,size-1
					C(MOD(m,1)+1,l,k-1,j,MOD(i,1)+1) = (A(m-1,l,MOD(k,1)+1,j,MOD(i-1,1)+1) * B(MOD(m+1,1)+1,l,MOD(k-1,1)+1,j,MOD(i+1,1)+1) + MOD(INT(A(m-1,l,MOD(k,1)+1,j,MOD(i,1)+1)),5) + B(MOD(m,1)+1,l-1,MOD(k+1,1)+1,j,MOD(i,1)+1)**2. ) + C(MOD(m,1)+1,l+1,k-1,j-1,MOD(i,1)+1) - C(MOD(m,1)+1,l,k-1,j,MOD(i,1)+1)!** B(MOD(INT(A(m,l,MOD(k,1)+1,j,MOD(i,1)+1)),1)+1,l,MOD(k,1)+1,j,MOD(i,1)+1)		
				enddo
					A(m-1,l,:,j,:) = REAL(B(:,l,:,j,1)) + INT(C(:,l,k,j,:))
			enddo
			B(MOD(m,1)+1,l-1,MOD(k,1)+1,j+1,MOD(i-1,1)+1) = 1
		enddo
	enddo
enddo
enddo

time = omp_get_wtime() - time

print *,"TIME",time/DBLE(repetition)

DEALLOCATE(B)

END PROGRAM

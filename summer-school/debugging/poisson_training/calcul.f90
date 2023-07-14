!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -*- Mode: F90 -*- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! calcul.f90  --- Calcul de la solution u a l'iteration n+1.
!!
!! Auteur          : Isabelle DUPAYS (CNRS/IDRIS - France)
!!                   <Isabelle.Dupays@idris.fr>
!!                   Philippe Parnaudeau (CNRS/IDRIS - France)
!!                   <Philippe.Parnaudeau@idris.fr>
!! Cree le         : aout 2007
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!*******************************************************************
SUBROUTINE calcul( u, u_temp, f, nx, ny, hx, hy, xnormer1)
  !*******************************************************************

  IMPLICIT NONE

  !--Declaration des variables

  !Nombre de points interieurs du sous domaine selon x et y
  INTEGER                                 :: nx,ny
  !Solution u a l'instant n
  REAL(kind=8), DIMENSION(0:nx+1, 0:ny+1) :: u
  REAL(kind=8), DIMENSION(0:nx+1, 0:ny+1) :: u_temp
  !Second membre
  REAL(kind=8), DIMENSION(1:nx, 1:ny)     :: f
  !Coefficients
  REAL(kind=8)                            :: hx,hy
  !Compteur
  INTEGER                                 :: i,j
  REAL(kind=8)                            :: xnormer1,r
  REAL(kind=8)                            :: c0,c1,c2


  !*******************************************************************
  !JACOBI USE every time --------------------------------------------- 
  !*******************************************************************
  c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
  c1 = 1./(hx*hx)
  c2 = 1./(hy*hy)
  
  !$OMP PARALLEL DO PRIVATE(i,j) &
  !$OMP SHARED (ny,nx,u,u_temp,f,c0,c1,c2) 
  do j= 1,ny
     do i= 1,nx
     u_temp(i,j)= c0 * (  c1*(u(i+1,j)+u(i-1,j)) &
                        + c2*(u(i,j+1)+u(i,j-1)) - f(i,j))
     enddo
  enddo
  !$OMP END PARALLEL DO
  
  !$OMP PARALLEL DO PRIVATE(i,j) &
  !$OMP SHARED (ny,nx,u,u_temp) 
  do j= 1,ny
     do i= 1,nx
        u(i,j)=u_temp(i,j)
     enddo
  enddo
  !$OMP END PARALLEL DO

  !*******************************************************************
  !RESIDU every time     --------------------------------------------- 
  !*******************************************************************
  xnormer1=0.
  r=0.  

  !$OMP PARALLEL DO PRIVATE(i,j,r) &
  !$OMP SHARED (ny,nx,u,f,hx,hy) &
  !$OMP REDUCTION(+:xnormer1) 
  do j=1,ny
     do i=1,nx
        r = ( u(i+1,j)-2*u(i,j)+u(i-1,j) )/(hx**2)&
          + ( u(i,j+1)-2*u(i,j)+u(i,j-1) )/(hy**2)&
          - f(i,j)
        xnormer1=r*r+xnormer1 
     enddo
  enddo
  !$OMP END PARALLEL DO

  
END SUBROUTINE calcul

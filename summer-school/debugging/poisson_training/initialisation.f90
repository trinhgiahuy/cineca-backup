!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -*- Mode: F90 -*- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! initialisation.f90  --- Initialisation des valeurs.
!!                 Calcul de la solution exacte u_exact et du second membre f.
!!
!! Auteur          : Isabelle DUPAYS (CNRS/IDRIS - France)
!!                   <Isabelle.Dupays@idris.fr>
!                    Philippe Parnaudeau (CNRS/IDRIS - France)
!                    <Philippe.Parnaudeau@idris.fr>
!  Cree le         : aout 2007
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!*******************************************************************
SUBROUTINE initialisation( comm2d, ndims, dims, u, u_exact,u_temp,&
                           f, nx, ny, hx, hy)
  !*******************************************************************

  USE MPI
  IMPLICIT NONE

  !--Declaration des variables

  !Nombre de points interieurs du sous domaine selon x et y
  INTEGER                                :: nx,ny
  !Nombre de points interieurs du domaine global suivant x et y
  INTEGER                                :: ntx,nty
  !Solution u a l'iteration n
  REAL(kind=8), DIMENSION(0:nx+1, 0:ny+1):: u
  REAL(kind=8), DIMENSION(0:nx+1, 0:ny+1):: u_temp
  !Solution exacte
  REAL(kind=8), DIMENSION(1:nx, 1:ny)    :: u_exact
  !Second membre
  REAL(kind=8), DIMENSION(1:nx, 1:ny)    :: f
  !Numerotation globale en x
  REAL(kind=8), DIMENSION(1:nx)          :: x
  !Numerotation globale en y
  REAL(kind=8), DIMENSION(1:ny)          :: y
  !Pas en x et en y
  REAL(kind=8)                           :: hx, hy
  !Coefficients
  REAL(kind=8)                           :: c0,c1, c2
  !Compteurs
  INTEGER                                :: i,j
  !Constantes MPI
  INTEGER                                :: comm2d, code
  INTEGER, DIMENSION(2)                  :: dims
  INTEGER, DIMENSION(2)                  :: coords
  INTEGER                                :: ndims, rang_ds_topo

  !*******************************************************************

  !Initialisation de u,u_exacte,f,x,y
  u(:,:)         = 0.d0
  u_temp(:,:)    = 0.d0
  u_exact(:,:)   = 0.d0
  f(:,:)         = 0.d0
  x(:)           = 0.d0
  y(:)           = 0.d0

  !Connaitre mon rang dans la topologie
  CALL MPI_COMM_RANK(comm2d, rang_ds_topo, code)

  !Connaitre mes coordonnees dans la topologie
  CALL MPI_CART_COORDS(comm2d, rang_ds_topo, ndims, coords, code)

  !Nombre de points du domaine global suivant x
  ntx = nx*dims(1)

  !Nombre de points du domaine global suivant y
  nty = ny*dims(2)

  !Calcul du pas en x et en y
  hx = 1./real(ntx+1)
  hy = 1./real(nty+1)

  !Calcul des coefficients
  c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
  c1 = 1./(hx*hx)
  c2 = 1./(hy*hy)

  !Calcul des valeurs de x(i) dans chaque sous-domaine
  DO i=1, nx
     x(i) = real(i + coords(1)*nx)*hx
  END DO

  !Calcul des valeurs de y(j) dans chaque sous-domaine
  DO j=1, ny
     y(j) = real(j + coords(2)*ny)*hy
  END DO

  !Calcul du second membre et de la solution exacte
  DO j=1,ny
     DO i=1,nx
        f(i,j) = 2*(x(i)*x(i)-x(i)+y(j)*y(j)-y(j))
        u_exact(i,j) = x(i)*y(j)*(x(i)-1)*(y(j)-1)
            END DO
     END DO

  u(1:nx,1:ny) = u_exact(1:nx,1:ny)*0.010+ u_exact(1:nx,1:ny)

END SUBROUTINE initialisation

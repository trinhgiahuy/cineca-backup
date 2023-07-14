!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -*- Mode: F90 -*- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! voisinage.f90  --- Calcul pour chaque processus de ses 4 processus voisins.
!!
!! Auteur          : Isabelle DUPAYS (CNRS/IDRIS - France)
!!                   <Isabelle.Dupays@idris.fr>
!                    Philippe Parnaudeau (CNRS/IDRIS - France)
!                    <Philippe.Parnaudeau@idris.fr>
!   Cree le         : aout 2007
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!*******************************************************************
SUBROUTINE voisinage(comm2d, voisin)
  !*******************************************************************

  USE MPI
  IMPLICIT NONE

  !--Declaration des variables
  INTEGER, PARAMETER             :: NB_VOISINS = 4
  INTEGER, PARAMETER             :: N=1, E=2, S=3, W=4
  INTEGER, DIMENSION(NB_VOISINS) :: voisin
  INTEGER                        :: comm2d,code

  !*******************************************************************

  !Initialisation du tableau voisin
  voisin(:) = MPI_PROC_NULL

  !Recherche des voisins Ouest et Est
  CALL MPI_CART_SHIFT( comm2d, 0, 1, voisin(W), voisin(E), code)

  !Recherche des voisins Sud et Nord
  CALL MPI_CART_SHIFT( comm2d, 1, 1, voisin(S), voisin(N), code)

END SUBROUTINE voisinage

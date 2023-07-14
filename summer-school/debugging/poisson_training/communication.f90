!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -*- Mode: F90 -*- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! communication.f90  --- Communication des valeurs aux interfaces pour
!!                        la solution u a l'iteration n.
!!
!! Auteur          : Isabelle DUPAYS (CNRS/IDRIS - France)
!!                   <Isabelle.Dupays@idris.fr>
!                     Philippe Parnaudeau (CNRS/IDRIS - France)
!                     <Philippe.Parnaudeau@idris.fr>
!! Cree le         : aout 2007
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!*******************************************************************
SUBROUTINE communication ( u, nx, ny, type_ligne, voisin, comm2d)

  !*****************************************************************

  USE MPI
  IMPLICIT NONE
  !--Declaration des variables

  !Nombre de points interieurs du sous domaine selon x et y
  INTEGER                                :: nx,ny
  !Solution u
  REAL(kind=8), DIMENSION(0:nx+1,0:ny+1) :: u
  !Type type_ligne
  INTEGER                                :: type_ligne
  !Tableau contenant les voisins du sous domaine courant
  INTEGER, PARAMETER                     :: NB_VOISINS = 4
  INTEGER, PARAMETER                     :: N=1, E=2, S=3, W=4
  INTEGER, DIMENSION(NB_VOISINS)         :: voisin
  !Constantes MPI
  INTEGER                                :: code, comm2d
  INTEGER, PARAMETER                     :: etiquette=100
  INTEGER, DIMENSION(MPI_STATUS_SIZE)    :: statut
  INTEGER                                :: local_rank, local_code

  !*****************************************************************

  CALL MPI_COMM_RANK(MPI_COMM_WORLD, local_rank, local_code)
  !Envoi au voisin N et reception du voisin S
  CALL MPI_SENDRECV( u(1,ny),   nx, MPI_DOUBLE_PRECISION,voisin(N),etiquette, &
                     u(1,0),    nx, MPI_DOUBLE_PRECISION,voisin(S),etiquette, &
                     comm2d, statut, code)

  !Envoi au voisin S et reception du voisin N
  ! deadlock simulation
  if ( local_rank .eq. 0)  then
    !print*," Remove this line "
    call mpi_ssend(u(1,1),1,mpi_real,local_rank+1,0,mpi_comm_world,code)
  else
  CALL MPI_SENDRECV( u(1,1),    nx, MPI_DOUBLE_PRECISION,voisin(S),etiquette, &
                     u(1,ny+1), nx, MPI_DOUBLE_PRECISION,voisin(N),etiquette, &
                     comm2d, statut, code)
  endif

  !Envoi au voisin W et  reception du voisin E en utilisant
  !le type predefini type_ligne
  CALL MPI_SENDRECV( u(1,1),     1, type_ligne,          voisin(W),etiquette, &
                     u(nx+1,1),  1, type_ligne,          voisin(E),etiquette, &
                     comm2d, statut, code)

  !Envoi au voisin E et  reception du voisin W en utilisant
  !le type predefini type_ligne
  CALL MPI_SENDRECV( u(nx,1),    1, type_ligne,          voisin(E),etiquette, &
                     u(0,1),     1, type_ligne,          voisin(W),etiquette, &
                     comm2d, statut, code)

END SUBROUTINE communication

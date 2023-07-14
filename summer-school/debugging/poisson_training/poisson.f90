!**********************************************************************
!   poisson.f90 - Resolution de l'equation de Poisson 
!   utilisant Jacobi sur le domaine [0,1]x[0,1] par une methode
!   aux differences finies,
!   avec comme solveur Jacobi.
!   Delta u = f(x,y)= 2*(x*x-x+y*y -y)
!   u sur les bords vaut 0
!   La solution exacte est u= x*y*(x-1)*(y-1)
!  
!   La valeur de u a l'iteration n+1 est donnee par la formule:
!   u(i,j)(n+1)= c0* (c1*(u(i+1,j)(n)+u(i-1,j)(n)) +
!                     c2*(u(i,j+1)(n)+u(i,j-1)(n)) - f(i,j))
!   avec
!   c0 = (0.5*hx**2*hy**2)/(hx**2+hy**2)
!   c1 =  1/hx**2
!   c2 =  1/hy**2
!   hx : pas en x
!   hy : pas en y
!
!   Pour simplifier la resolution, on se donne la taille locale de chacun
!   des sous domaines en x et y, chacun des sous domaines a le meme
!   nombre de points interieurs en x (nx) et en y (ny).
!   Ces valeurs seront lues dans un fichier d'entree : poisson.data
!
!   Pour chaque processus :
!   1) decomposer le domaine
!   2) connaitre ses 4 voisins
!   3) echanger les points aux interfaces
!   4) calculer
!****************************************************************************

PROGRAM poisson
  USE MPI
  USE MPI_TIMES
  IMPLICIT  NONE

  !---Declaration des variables

  !Solution u et u_nouveau a l'iteration n et n+1
  REAL(kind=8), DIMENSION(:,:), ALLOCATABLE :: u,u_temp
  !Solution exacte
  REAL(kind=8), DIMENSION(:,:), ALLOCATABLE :: u_exact
  !Second membre
  REAL(kind=8), DIMENSION(:,:), ALLOCATABLE :: f
  !Second membre
  real,pointer::p(:)=>null()

  !Nombre de points du sous domaine selon x et y
  INTEGER                                   :: nx,ny
  INTEGER                                   :: rang !Numero du sous domaine
  INTEGER                                   :: nb_procs !Nombre de processus

  !Tableau contenant les voisins du sous domaine courant
  INTEGER, PARAMETER                        :: NB_VOISINS = 4
  INTEGER, DIMENSION(NB_VOISINS)            :: voisin
  INTEGER, PARAMETER                        :: N=1, E=2, S=3, W=4
  INTEGER                                   :: it !Nombre iterations en temps

  !Nombre iterations maximum en temps
  INTEGER                                   :: it_max
  REAL(kind=8)                              :: prec
  REAL(kind=8)                              :: diffnorm,diffnorm_temp
  REAL(kind=8)                              :: erreur,hx,hy
  LOGICAL                                   :: convergence

  !Constantes MPI
  INTEGER                                   :: comm2d,code,type_ligne
  INTEGER, PARAMETER                        :: ndims = 2
  INTEGER, DIMENSION(2)                     :: dims
  LOGICAL, DIMENSION(2)                     :: periods
  LOGICAL, PARAMETER                        :: reorganisation = .false.
  !****************************************************************************
 

  !Initialisation de MPI
  CALL MPI_INIT(code)

  !Savoir quel processus je suis
  CALL MPI_COMM_RANK( MPI_COMM_WORLD, rang, code)

  !Connaitre le nombre total de processus
  CALL MPI_COMM_SIZE( MPI_COMM_WORLD, nb_procs, code)

  OPEN  (10, FILE='poisson.data', STATUS='OLD')
  READ  (10,*) nx
  READ  (10,*) ny
  READ  (10,*) it_max
  READ  (10,*) prec
  CLOSE (10)

  if (rang == 0) then
      print *,'nx :',nx  
      print *,'ny :',ny
      print *,'it_max :',it_max
      print *,'prec :',prec
   end if

  !Allocation dynamique des tableaux utilises dans le  calcul
  ! u, u_nouveau, u_exact, f
  ALLOCATE (u(0:nx+1,0:ny+1))
  ALLOCATE (u_temp(0:nx+1,0:ny+1))
  ALLOCATE (u_exact(1:nx,1:ny),f(1:nx,1:ny))

  !Connaitre le nombre de processus selon x et le nombre de processus
  !selon y en fonction du nombre total de processus nb_procs
  dims(2) = 0
  SELECT CASE(nb_procs)
     CASE(2048)
        dims(1)=64
     CASE(1024)
        dims(1)=64
     CASE(512)
        dims(1)=32
     CASE(256)
        dims(1)=16   
     CASE(128)
        dims(1)=16
     CASE(64)
        dims(1)=8
     CASE(32)
        dims(1)=8
     CASE(16)
        dims(1)=4
     CASE DEFAULT
        dims(1)=0
  END SELECT

  !p(1)=0.
  CALL MPI_DIMS_CREATE( nb_procs, ndims, dims, code )
  if (rang == 0) print *,'Nombre de domaines en x ', dims(1),&
       'Nombre de domaines en y ', dims(2)

  !Creation de la grille de processus 2D
  periods(:) = .false.

  CALL MPI_CART_CREATE( MPI_COMM_WORLD, ndims, dims, periods,reorganisation, &
                        comm2d, code )

  !Initialisation des valeurs
  CALL initialisation( comm2d, ndims, dims, u, u_exact,u_temp, &
                       f, nx, ny, hx, hy)
  
  !Recherche de ses 4 voisins pour chaque processus
  CALL voisinage(comm2d,voisin)

  PRINT *, "Processus ",rang,":",voisin

  !Creation du type type_ligne pour echanger les points a l'ouest et a l'est
  CALL MPI_TYPE_VECTOR( ny, 1, nx+2, MPI_DOUBLE_PRECISION, type_ligne, code )
  CALL MPI_TYPE_COMMIT( type_ligne, code )

  !Schema iteratif en temps
  it = 0
  convergence = .false.

    
  !Mesure du temps en seconde a l'entree de la boucle
  CALL MPI_TIME(0)

  DO WHILE ((.NOT. convergence) .AND. (it < it_max) )

     it = it +1

     !Echange des points aux interfaces pour  u a l'iteration n
     CALL communication (u, nx, ny, type_ligne, voisin, comm2d)

     !Calcul de u a l'iteration n+1
     CALL calcul( u, u_temp, f, nx, ny, hx, hy, erreur)

     CALL MPI_ALLREDUCE( erreur, diffnorm_temp, 1, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, comm2d, code )
   
     diffnorm = SQRT(diffnorm_temp/(nb_procs*nx*ny))
 
     !Arret du programme si on a atteint la precision machine obtenu
     !par la fonction F90 EPSILON
     convergence = (diffnorm < prec) .OR. (diffnorm > 1E20)

  END DO

  !Mesure du temps en seconde a la sortie de la boucle
  CALL MPI_TIME(1)

  IF ((rang == 0)) THEN
      PRINT 10, it, diffnorm
10    FORMAT  ('it = ',I6,' le residu est ',f12.8)

  !Affichage pour le processus 0 de la difference
     PRINT *,'Solution exacte u_exact ', 'Solution calculee u'
     PRINT 12, u_exact(1,1),  u (1,1)
     PRINT 13, u_exact(nx,ny),  u (nx,ny)
12   FORMAT  ('u_exact(1,1)=  ',f12.8,' u(1,1) =  ',f12.8)
13   FORMAT  ('u_exact(nx,ny)=  ',f12.8,' u(nx,ny)=  ',f12.8)
  END IF

  !Desallocation des tableaux u,u_exact et f
  DEALLOCATE(u, u_exact, f)

  !Liberation du type type_ligne et du communicateur comm2d
  CALL MPI_TYPE_FREE( type_ligne, code )
  CALL MPI_COMM_FREE( comm2d, code )

  !Desactivation de MPI
  CALL MPI_FINALIZE(code)


END PROGRAM poisson

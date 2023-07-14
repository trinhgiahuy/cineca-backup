FC = mpif90
LD = $(FC)
FFLAGS = -g -O2   
LDFLAGS = -g -O2  
INCS = -I.
#LIBS = /bgp/usermisc/hhhibm11/libraries/libmpitrace/libmpitrace.a
CPP = cpp
CPPFLAGS = 
 
OBJS =  MPI_Time.o poisson.o calcul.o communication.o initialisation.o voisinage.o

.SUFFIXES: .o .f90 

all: poisson.exe

poisson.exe : $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	rm *.mod *.o

.f90.o:
	$(FC) $(INCS) -c $(FFLAGS) $<


clean :
	rm -f poisson.exe *.mod *.o *.lst

cleanall : clean
	rm -f core

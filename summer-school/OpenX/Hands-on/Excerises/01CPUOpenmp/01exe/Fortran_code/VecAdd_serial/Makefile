FC=gfortran 
FFLAGS=-O3 -Wall -Wextra -std=f2008

SRC=vec_add.f90
OBJ=${SRC:.f90=.o}

%.o: %.f90 
	$(FC) $(FFLAGS) -o $@ -c $<

output: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)
	
clean: 
	@rm -f *.o output

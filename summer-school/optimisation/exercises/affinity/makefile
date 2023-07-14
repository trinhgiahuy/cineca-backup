#
FFLAGS=-qopenmp

FC=ifort
CC=icc

FOBJ=affinity.o
COBJ=mysecond.o

affinity: $(FOBJ) $(COBJ)
	$(FC) -o $@ $(FFLAGS) $(FOBJ) $(COBJ)

$(COBJ):mysecond.c
	$(CC) -c  $<

$(FOBJ):affinity.f90
	$(FC) -c $(FFLAGS) $<

clean:
	rm *.o

#
FC=ifort
CC=icc
FFLAGS=-g -O2 -qopenmp

stream: stream.o mysecond.o
	$(FC) $(FFLAGS) -o $@ stream.o mysecond.o

triad: triad.f90 mysecond.o
	$(FC) $(FLAGS) -o $@ triad.o mysecond.o

stream.o:stream.f
	$(FC) $(FFLAGS) -c $<

mysecond.o:mysecond.c
	$(CC) $(CFLAGS) -c $<

clean:
	-rm *.o stream triad

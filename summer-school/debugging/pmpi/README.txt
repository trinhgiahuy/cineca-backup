mpiifort -shared -fpic -o libmysend.so mysend.f90
mpirun -genv LD_PRELOAD ./libmysend.so -np 2 ./example

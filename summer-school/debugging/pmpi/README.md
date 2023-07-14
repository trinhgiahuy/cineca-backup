# Create and MPI library with PMPI example

```bash
# compile original program
mpiifort -o example example.f90
```

You can either:
1. re-link the program with your version
2. use LD_PRELOAD (mpirun)

```bash
1.
mpiifort -c mysend.f90
mpiifort -o example exmaple.o mysend.o

2.
# create a shared library with my own version of mpi_send
mpiifort -shared -fpic -o libmysend.so mysend.f90

#run mpirun with LD_PRELOAD
mpirun -genv LD_PRELOAD ./libmysend.so -np 2 ./example
```

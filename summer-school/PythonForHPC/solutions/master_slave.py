#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import glob
import time
from subprocess import Popen
from mpi4py import MPI

worktag = 0 # sent from the master to the slave when imposing to work
freetag = 1 # sent from the slave to the master to tell that it's free
dietag  = 2 # sent from the master to the slave when killing the slave

def input_file(my_path):
    return glob.glob(my_path+'/input*.txt')

def run(executable, inputs):
    for inp in inputs:
        with open(inp+'.out', 'w') as out_file:
           pid = Popen([executable, inp], stdout=out_file)
           pid.wait()

def master(remaining_inputs):
    # the first inputs are sent apart (this avoids deadlock)
    for i in range(1,size):
        comm.send(obj=remaining_inputs.pop(), dest=i, tag=worktag )

    # as long as there are still input to process, they are sent to the
    # slaves as soon as they communicate to the master that they are free
    while remaining_inputs:
        free_rank = comm.recv(source=MPI.ANY_SOURCE, tag=freetag)
        comm.send(obj=remaining_inputs.pop(), dest=free_rank, tag=worktag )

    # when all the inputs have been distributed, the next times the slaves
    # communicate being free, the master send the dietag to them
    for i in range(1,size):
        free_rank = comm.recv(source=MPI.ANY_SOURCE, tag=freetag)
        comm.send(obj=None, dest=free_rank, tag=dietag)
#        print("Killing rank {:d}".format(free_rank))

def slave():
    slave_status=MPI.Status() # inizialization

    # the slave receive its first input from the master 
    input_to_elaborate = comm.recv(source=0, tag=worktag, status=slave_status)

    # as long as the slave is ordered to work (i.e. worktag) it keeps running inputs
    # (thus it will stop when receiving a dietag for instance)
    inputs_counter = 0
    while slave_status.Get_tag() == worktag:
        run(executable, [input_to_elaborate])
#        print("I'm the slave with rank {:d} and I worked on the input {:s}".format(rank, input_to_elaborate))
        inputs_counter += 1

        # the slave tells to the master that it's free
        comm.send(obj=rank, dest=0, tag=freetag)
        # the slave waits for another inputs (but it can receive either a freetag
        # or a dietag)
        input_to_elaborate = comm.recv(source=0, tag=MPI.ANY_TAG, status=slave_status)

    return inputs_counter

if __name__ == '__main__':
    executable = os.path.abspath(sys.argv[1])
    input_dir = os.path.abspath(sys.argv[2])
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        inputs = input_file(input_dir)
        inputs.sort()
        print("Number of elaborations: {:d}, running on {:d} MPI tasks".format(len(inputs), size))
        master(inputs)
    else:
        inputs_counter = slave()
        t3 = time.time()
        print("I am rank {:d}, elapsed time {:5.2f}, executed {:d} elaborations.".format(rank, t3-t1, inputs_counter))


    comm.Barrier()
    if rank == 0:
        t2 = time.time()
        print("Elapsed time {:5.2f}".format(t2-t1))

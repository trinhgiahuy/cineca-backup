#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import glob
import time
from subprocess import Popen
from mpi4py import MPI

def input_file(my_path):
    return glob.glob(my_path+'/input*.txt')

def run(executable, inputs):
    for inp in inputs:
        with open(inp+'.out', 'w') as out_file:
           pid = Popen([executable, inp], stdout=out_file)
           pid.wait()

if __name__ == '__main__':
    executable = os.path.abspath(sys.argv[1])
    input_dir = os.path.abspath(sys.argv[2])
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    inputs = input_file(input_dir)
    if rank == 0:
        print("Number of elaborations: {:d}, running on {:d} MPI tasks".format(len(inputs), size))

    inputs.sort()
    queue = []

    comm.Barrier()
    t1 = time.time()

    for i in range(rank ,len(inputs), size):
        queue.append(inputs[i])
    print("I am rank {:d}, elaborations {:d}".format(rank, len(queue)))

    run(executable, queue)

    t3 = time.time()
    print("I am rank {:d}, elapsed time {:5.2f}".format(rank, t3-t1))

    comm.Barrier()
    t2 = time.time()
    if rank == 0:
        print("Elapsed time {:5.2f}".format(t2-t1))
    

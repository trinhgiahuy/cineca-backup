#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import time
from random import randint
from subprocess import Popen


def some_condition():
    rn = randint(1,20)
    if rn < 20:
        return True
    else:
        return False


def check_for_fault(nodes):
    """This function checks if there are fault nodes."""
    ok_nodes = set()
    for node in nodes:
        if some_condition(): 
            ok_nodes.add(node)
        else:
            sys.stderr.write("Fault on node {}.\n".format(node))
    if not ok_nodes:
        sys.stderr.write('ERROR: All nodes fault.\n')
        exit(1)
    return ok_nodes


def run(elaborations, nodes):
    machinefile = 'machinefile.txt'
    for elab in elaborations:
        nodes = check_for_fault(nodes)
        with open(machinefile, 'w') as mf:
            for node in nodes:
                #mf.write(node.strip()+' slots=36\n') # use with OpenMPI
                mf.write(node.strip()+':36\n') # use with IntelMPI
        time.sleep(1)
        pid = Popen(['mpirun', '-machinefile', machinefile, mpi_program, str(elab)])
        pid.wait()

# main
if __name__ == '__main__':
    mpi_program = os.path.abspath(sys.argv[1])
    input_file = os.path.abspath(sys.argv[2])
    nodefile = 'nodefile.txt'

    # create a list of cores/nodes
    cores = []
    with open(nodefile, 'r') as nf:
        for line in nf:
            cores.append(line.strip())
    # create a set of nodes
    nodes = set(cores)

    # create a list of elaborations from the input file
    elaborations = []
    with open(input_file, 'r') as inf:
        for line in inf:
            elaborations.append(line.strip())

    run(elaborations, nodes)
    
    print('FINITO!')


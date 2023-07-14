#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Serial program that runs a queue of serial simulations.
Usage: ./serial_queue.py <executable> <input_directory>
"""
from __future__ import print_function
import os
import sys
import glob
import time
from subprocess import Popen

def input_files(my_path):
    return glob.glob(my_path+'/input*.txt')

def run(executable, inputs):
    for inp in inputs:
        with open(inp+'.out', 'w') as out_file:
           pid = Popen([executable, inp], stdout=out_file)
           pid.wait()

if __name__ == '__main__':
    executable = os.path.abspath(sys.argv[1])
    input_dir = os.path.abspath(sys.argv[2])

    inputs = input_files(input_dir)
    print("Number of elaborations: {:d}".format(len(inputs)))

    t1 = time.time()
    run(executable, inputs)
    t2 = time.time()

    print("Elapsed time {:5.2f}".format(t2-t1))
    

import os
import sys
import random

if __name__ == '__main__':
    os.mkdir('inputs')
    for i in range(int(sys.argv[1])):
        with open('inputs/input{:04d}.txt'.format(i), 'w') as f:
            f.write('{:.3f} # seconds to wait\n'.format(random.random()))
    

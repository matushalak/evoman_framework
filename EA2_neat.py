###############
# @matushalak
# code adapted from NEAT-python package
###############
import multiprocessing
import os

import neat

def eval_genome():
    pass

# Run for up to 300 generations.
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    #winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    #print('\nBest genome:\n{!s}'.format(winner))
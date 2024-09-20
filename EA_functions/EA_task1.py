# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# import self-made packages
from EA_functions.EA_task1 import EA_task1_function


def EA_task1_function(dom_l, dom_u, npop, n_vars):
    # steps:

    # initialize population. TO DO: add option to use previous populations
    population, fitness_pop = initialize_pop(dom_l, dom_u, npop, n_vars)

    # ...

    # perform iterations (go through the generations)
    for i in range(max_gen):

        # initialize next gen:
        parents, fitness_parents = initialize_next_gen()

        # keep the best parent profile? Elitism

        # make children: first perform crossover 
        children, fitness_children = crossover()

        #make children: perform mutation
        children, fitness_children = mutation()

        # update generation (replacement)

        # ...

        # ...

        # ...

    # final best result:


def initialize_pop(dom_l, dom_u, npop, n_vars):
    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))


    
# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

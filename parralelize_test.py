from evoman.environment import Environment
from demo_controller import player_controller

import multiprocessing as mp
from joblib import Parallel, delayed
import numpy as np

def initialize_population(popsize, individual_dims, gene_limits:list[float, float]):
    ''' Generate a population of 
        N = popsize solutions, each solution containing
        N = individual_dims genes, each gene within <gene_limits>'''
    population = np.random.uniform(*gene_limits, (popsize, individual_dims))
    return population.tolist()

pop = initialize_population(100, 100, [-1,1])

# # runs game (evaluate fitness for 1 individual)
def run_game(env:Environment,individual, test=False):
    '''Runs game and returns individual solution's fitness'''
    # vfitness, vplayerlife, venemylife, vtime
    if isinstance(individual, float):
        breakpoint()
    fitness ,p,e,t = env.play(pcont=individual)
    if test == False:
        return fitness
    else:
        return fitness ,p,e,t

def evaluate_fitnesses(env, population):
    ''' Evaluates fitness of each individual in the population of solutions
    parallelized for efficiency'''
    # Instead of passing the full environment, pass only the configuration or parameters needed to reinitialize it
    name  = env.experiment_name
    contr = env.player_controller
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(name, contr, ind) for ind in population
    )
    return fitnesses

def run_game_in_worker(name, contr, ind):
    # Recreate or reinitialize the environment from env_config inside the worker
    env = Environment(experiment_name=name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=contr, # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    return run_game(env, ind)
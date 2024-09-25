# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

# import self-made packages
#from EA_functions.EA_task1 import EA_task1_function


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'fluctuation_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[4],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)


    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    
        
    # settings:
    dom_u = 1
    dom_l = -1
    npop = 100
    max_gen = 100
    mutation_rate = 0.2
    last_best = 0
    settings = (dom_u, dom_l, npop, max_gen, mutation_rate, last_best, n_vars)

    # call the EA function
    EA_task1_function(settings, env)


def EA_task1_function(settings, env):
    
    # unpack settings tuple
    dom_u, dom_l, npop, max_gen, mutation_rate, last_best, n_vars = settings

    # steps: 
    #   - initialize population
    #   - iterate through generations
    #       - select parents for producing next generation
    #       - perform crossover
    #       - perform mutation

    # initialize population. TO DO: add option to use previous populations
    population, fitness_pop, fitness_mean, fitness_std  = initialize_pop(dom_l, dom_u, npop, n_vars, env)
    best_index = np.argmax(fitness_pop)

    # perform iterations (go through the generations)
    for i in range(max_gen):

        # initialize next gen:
        parents = initialize_next_gen(population, fitness_pop)
        #print(parents.shape)

        # keep the best parent profile? Elitism --> later

        # make children: first perform crossover 
        children = crossover(parents)

        #make children: perform mutation
        children = mutation(children, mutation_rate)

        # evaluate fitness
        fitness_children = evaluate(env, children)
        
        # Elitism: Replace the worst individual in the children with the best from the previous generation
        worst_index = np.argmin(fitness_children)
        children[worst_index] = population[best_index]
        fitness_children[worst_index] = fitness_pop[best_index]

        # update generation (replacement)
        population = children
        fitness_pop = fitness_children


        best_index = np.argmax(fitness_pop)
        #best_individual = population[best_index]
        #print(f'Best indivual array: {best_individual}')
        fitness_mean = np.mean(fitness_pop)
        print(f"Generation: {i}, fitness: {fitness_pop[best_index]}, mean {fitness_mean}")

    # final best result:


def initialize_pop(dom_l, dom_u, npop, n_vars, env):

    population = np.random.uniform(dom_l, dom_u, (npop, n_vars)) # copy-paste from opt_speci_demo

    fitness_pop = evaluate(env, population)
    solutions = [population, fitness_pop]
    env.update_solutions(solutions)

    # calculate mean and std
    fitness_mean = np.mean(fitness_pop)
    fitness_std = np.std(fitness_pop)

    return population, fitness_pop, fitness_mean, fitness_std


def initialize_next_gen(population, fitness, tour_size=3):

    '''This function uses tournament selection'''
    winner = []
    for _ in range(len(population)):
        participants_idx = np.random.choice(len(population), tour_size, replace=False)
        best_idx = participants_idx[np.argmax(fitness[participants_idx])]
        winner.append(population[best_idx])

    return np.array(winner)


def crossover(parents, crossover_rate=0.7):
    '''This function performs single point crossover'''
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        rand_value = np.random.rand()
        if rand_value < crossover_rate:
            point = np.random.randint(1, parent1.shape[0] - 1)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

        # append the offsprings    
        offspring.append(offspring1)
        offspring.append(offspring2)

    return np.array(offspring)


def mutation(population, mutation_rate):
   
    '''This function applies random mutation to all genes in individual'''

    for indi in population:
        for i in range(len(indi)):
            rand_value = np.random.rand()
            if rand_value < mutation_rate:
                indi[i] += np.random.normal()

    return population



    
# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))






if __name__ == '__main__':
    main()
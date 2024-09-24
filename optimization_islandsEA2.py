###############################################################################
# Author: Matus Halak       			                                      #
# matus.halak@gmail.com   
# 
# In this script I implement Speciation (ISLANDS MODEL) as a way to increase diversity
# GOALS: also implement uncorrelated mutation with N sigmas!!! (put also in basic script)
# Figure out how to use numpy arrays preferably
# save results with a function
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
import argparse
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from pandas import read_csv

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-pop', '--popsize', type=int, required=False, default = 100, help="Population size (eg. 100)")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 500, help="Max generations (eg. 500)")
    parser.add_argument('-cr', '--crossover_rate', type=float, required=False, default = 0.5, help="Crossover rate (e.g., 0.8)")
    parser.add_argument('-mr', '--mutation_rate', type=float, required=False, default = 0.1, help="Mutation rate (e.g., 0.05)")
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default = 10, help="Number of Hidden Neurons (eg. 10)")
    parser.add_argument('-tst', '--test', type=bool, required=False, default = False, help="Train or Test (default = Train)")
    parser.add_argument('-nme', '--enemy', type=int, required=False, default = 2, help="Select Enemy")

    return parser.parse_args()

def main():
    '''Main function for basic EA, runs the EA which saves results'''
    # command line arguments for experiment parameters
    args = parse_args()
    popsize = args.popsize
    mg = args.maxgen
    cr = args.crossover_rate
    mr = args.mutation_rate
    n_hidden = args.nhidden
    enemy = args.enemy

    if isinstance(args.exp_name, str):
        experiment_name = 'islands_' + args.exp_name
    else:
        experiment_name = 'islands_' + input("Enter Experiment (directory) Name:")
    
    # add enemy name
    experiment_name = experiment_name + f'_{enemy}'
    # directory to save experimental results
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[enemy],
                    playermode="ai",
                    player_controller=player_controller(n_hidden), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
    
    # default environment fitness is assumed for experiment
    env.state_to_log() # checks environment state

    if args.test == True:
        best_solution = np.loadtxt(experiment_name+'/best.txt')
        print( '\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed','normal')
        vfitness, vplayerlife, venemylife, vtime = run_game(env, best_solution, test = True)
        print('vfitness, vplayerlife, venemylife, vtime:\n',
              vfitness, vplayerlife, venemylife, vtime)
        sys.exit(0)

    if not os.path.exists(experiment_name+'/evoman_solstate'):
        print( '\nNEW EVOLUTION\n')
        # with initialization
        evol_exp = islands(popsize, mg, mr, cr, n_hidden, experiment_name,
                            env, n_islands=5, gen_per_island=10)

# for parallelization later
# worker_env = None
def initialize_env(name, contr, enemies):
    enviro =  Environment(experiment_name=name,
                        enemies=enemies,
                        playermode="ai",
                        player_controller=contr, # you  can insert your own controller here
                        enemymode="static",
                        level=2,
                        speed="fastest",
                        visuals=False)
    return enviro

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
    enemies = env.enemies
    fitnesses = Parallel(n_jobs= -1)(
        delayed(run_game_in_worker)(name, contr, enemies, population[ind,:]) for ind in range(population.shape[0]))
    return np.array(fitnesses)
    # return np.array([run_game(env, population[ind,:]) for ind in range(population.shape[0])])

def run_game_in_worker(name, contr, enemies, ind):
    # Recreate or reinitialize the environment from env_config inside the worker
    # global worker_env
    # if worker_env is None:
    worker_env = initialize_env(name, contr, enemies)
    return run_game(worker_env, ind)

# check and apply limits on genes (of offspring - after mutation / recombination)
def within_genetic_code(gene, gene_limits:list[float, float]):
    ''' Function that verifies whether a given gene in an individual 
    is within the specified gene limits'''
    if gene <= gene_limits[0]:
        return gene_limits[0]
    elif gene >= gene_limits[1]:
        return gene_limits[1]
    else:
        return gene

# initializes random population of candidate solutions
def initialize_population(popsize, individual_dims, gene_limits:list[float, float]):
    ''' Generate a population of 
        N = popsize solutions, each solution containing
        N = individual_dims genes, each gene within <gene_limits>'''
    population = np.random.uniform(*gene_limits, (popsize, individual_dims))
    return population

# def speciation
def make_islands(n_islands:int, popsize:int, ind_size:int):
    if popsize % n_islands != 0 or popsize <= 0:
        popsize = int(input(f'Enter population size divisible by number of islands ({n_islands}):'))
    # islands = np_array (rows = islands, columns = individuals, depth = genes)
    return np.random.uniform(-1.0,1.0,(n_islands, popsize//n_islands, ind_size))
  
def island_life(island, generations_per_island,env, mr = .1, cr = .6):
    # island is a 3D array - 1(island) x popsize x n_genes
    popsize, indsize = island.shape[-2:]
    pop = np.reshape(island,(popsize, indsize)) # np 2D array popsize x n_genes
    fit = evaluate_fitnesses(env,pop) # np 1d array popsize
    
    # Start tracking best individual
    best_idx = np.argmax(fit) # indexes row of pop / fitness value of best individual
    best_individual = pop[best_idx,:] # np 1D array n_genes
    best_fitness = fit[best_idx]
    
    # stagnation prevention
    stagnation = 0
    starting_mutation_rate, starting_crossover_rate = mr, cr
    sigma_prime = .05
    for g in range(generations_per_island):
        # niching
        shared_fit = fitness_sharing(fit, pop) # np 1d array popsize
        # Parent selection: (Tournament? - just first try) Probability based - YES
        parents, parent_fitnesses = parent_selection(pop, shared_fit, env) # np 2d array (popsize x n_genes) & np 1d array popsize

        # crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
        offspring = crossover(parents, cr, blend_recombination)
        
        # mutation: Uncorrelated mutation with N step sizes
        offspring_mutated, sigma_primes = zip(*[uncorrelated_mut_one_sigma(offspring[ind,:], sigma_prime, mr) for ind in range(offspring.shape[0])])
        offspring_mutated = np.array(offspring_mutated)
        offspring_fitnesses = evaluate_fitnesses(env, offspring_mutated)
        sigma_prime = sigma_primes[np.argmax(offspring_fitnesses)]

        # Survivor selection with elitism & some randomness
        pop, fit = survivor_selection(parents, parent_fitnesses,
                                      list(offspring_mutated), offspring_fitnesses)
        
        # Check for best solution
        best_idx = np.argmax(fit)
        best_fitness = fit[best_idx]
        best_individual = pop[best_idx,:]
        
    np.reshape(pop, (1,popsize, indsize)) # back to island format
    return pop, fit, (best_individual, best_fitness)
        

def migration(islands, N:int = 5):
    '''Exhange 1/N of population between islands randomly
        by default exchange 1/3
        
        Using RING-ISLAND TOPOLOGY FOR NOW
        '''
    # print(f'Migration: {islands.shape}')
    migrated_islands = []
    # rows are individuals
    # vstack to that migration from last island to first completes circle
    for island1, island2 in zip(islands[:, :, :], np.vstack([islands[1:, :, :], islands[:1,:,:]])):
        # np.random.shuffle(island1)
        # np.random.shuffle(island2)
        # breakpoint()
        updated_island = island2.copy()
        # last part N of island 1 migrates to 1st part N of island2
        updated_island[:island2.shape[0]//N,:] = island1[-island2.shape[0]//N:,:]
        migrated_islands.append(updated_island)
    return migrated_islands

def redistribute_to_islands(population_together):
    ''''More advanced - redistribute evolved population to different islands
    eg. based on Euclidian Distance'''
    pass

# Define a wrapper for parallelization (joblib)
def parallel_island_life(island_slice, gen_per_island,
                         name, contr, enemies):
    # Reinitialize or avoid non-picklable objects inside this function if needed
    env = initialize_env(name, contr, enemies)  # Re-initialize the environment inside each worker if necessary
    return island_life(island_slice, gen_per_island, env)

def islands(popsize:int, max_gen:int, mr:float, 
            cr:float, n_hidden_neurons:int,
            experiment_name:str, env:Environment,
            n_islands:int, gen_per_island:int):
    '''Basically the big EA loop'''
    # number of weights for multilayer with 10 hidden neurons
    individual_dims = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    # initiation time
    ini = time.time()    
    # initialize
    islands = make_islands(n_islands, popsize, individual_dims)

    # for parallelization
    name  = env.experiment_name
    contr = env.player_controller
    enemies = env.enemies
    for cycle in range((max_gen//2)//gen_per_island):
        # Parallelize island life across islands
        results = Parallel(n_jobs=-1)(delayed(parallel_island_life)(islands[isl, :, :], gen_per_island, name, contr, enemies) 
                                      for isl in range(islands.shape[0]))
        
        # Unzip results into evolved islands and best individuals
        islands_evolved, fitnesses_evoled, islands_best = zip(*results)
        
        fitnesses_evoled = np.array(fitnesses_evoled)
        islands_evolved = np.array(islands_evolved)
        # breakpoint()
        islands_best = list(islands_best)
        best_fits = [ib[-1] for ib in islands_best]
        fin = time.time()
        print(f'Cycle {cycle} best fitness: {max(best_fits)}, mean fitness: {np.mean(fitnesses_evoled)},t:{fin-ini}')
        # migrate between islands
        np.random.shuffle(islands_evolved) # get rid of order
        islands = np.array(migration(islands_evolved)).reshape((n_islands,-1, individual_dims))

    return islands_best
    # run second half of generations in normal evolutionary mode (mixing everyone together)
    # (more advanced - put different individuals into different islands)
    # for gen in range(max_gen//2):
    #     # normal evolution
    #     pass

# promoting diversity
def fitness_sharing(fitnesses: list, population: list, 
                    gene_limits:list[float, float] = [-1.0,1.0], k = 0.15):
    """Apply fitness sharing as described in the slide to promote diversity and niche creation."""
    # Calculate the max possible distance between two genes
    gene_distance_max = gene_limits[1] - gene_limits[0]  
    # Calculate max possible distance between two individuals and sigma_share
    max_possible_distance = gene_distance_max * np.sqrt(population.shape[1]) 
    # Scaling factor k controls fitness sharing radius
    sigma_share = k * max_possible_distance  

    shared_fitnesses = []
    for i in range(population.shape[0]):
        niche_count = 0
        for j in range(population.shape[0]):
            if i != j:  # No need to compare with self
                distance = np.linalg.norm(population[i,:] - population[j,:]) 
                if distance < sigma_share:
                    niche_count += (1 - (distance / sigma_share))
        # Add the individual's own contribution (sh(d) = 1 for distance 0 with itself)
        niche_count += 1  
        shared_fitness = fitnesses[i] / niche_count
        shared_fitnesses.append(shared_fitness)
    return shared_fitnesses

# Tournament selection
def tournament_selection(population, fitnesses, k: int = 15) -> list:
    """Tournament selection using tournament size = k.
            Gives 1 WINNER of TOURNAMENT"""
    players = np.random.choice(np.arange(population.shape[0]), size=k)
    best_individual_idx = max(players, key=lambda i: fitnesses[i])
    return population[best_individual_idx,:]

# Parent selection
def parent_selection(population, fitnesses, env:Environment, n_children = 2):
    '''Tournament-based parent selection (for now)'''    
    n_parents = int(population.shape[0] / n_children)
    # genotypes of parents, fitnesses of those genotypes
    g_parents = []
    for _ in range(n_parents):
        for _ in range(n_children):
            parent = tournament_selection(population, fitnesses) # 1d np array n_genes
            g_parents.append(parent)
    # (parallelized) fitness evaluation
    g_parents = np.array(g_parents)
    f_parents = evaluate_fitnesses(env, g_parents) # 1d array
    return g_parents, f_parents

# Survivor selection with elitism and random diversity preservation
def survivor_selection(parents, parent_fitnesses, 
                       offspring, offspring_fitnesses, elite_fraction=0.8):
    """Select survivors using elitism with some randomness to maintain diversity."""
    # Combine parents and offspring
    total_population = np.vstack([parents, offspring])
    total_fitnesses = np.hstack([parent_fitnesses, offspring_fitnesses])
    # Sort by fitness in descending order
    # Sort np array?
    sorted_individuals = sorted(zip(total_fitnesses, total_population), key=lambda x: x[0], reverse=True)
    
    # Select elites (the top portion of the population)
    num_elites = int(elite_fraction * len(parents))  # Determine number of elites to preserve
    elites = sorted_individuals[:num_elites]
    # Select the remaining individuals randomly from the rest to maintain diversity
    remaining_individuals = sorted_individuals[num_elites:]
    np.random.shuffle(remaining_individuals)  # Shuffle to add randomness
    # Select the remaining individuals to fill the population
    num_remaining = len(parents) - num_elites
    selected_remaining = remaining_individuals[:num_remaining]
    
    # Combine elites and randomly selected individuals
    survivors = elites + selected_remaining
    # Separate fitnesses and individuals for the return
    return np.array([ind for _, ind in survivors]), np.array([fit for fit, _ in survivors])

def whole_arithmetic_recombination(p1:list, p2:list, weight:float = .57)->list[list,list]:
    ''''Apply whole arithmetic recombination to create offspring
    --> potentially switch to this once close to solution'''
    ch1 = [weight*x + (1-weight)*y for x,y in zip(p1,p2)]
    ch2 = [weight*y + (1-weight)*x for x,y in zip(p1,p2)]
    return ch1, ch2

def blend_recombination(p1: list[float], p2: list[float], alpha: float = 0.5) -> list[list[float], list[float]]:
    '''Apply blend recombination (BLX-alpha) to create two offspring.
    --> really explorative'''
    ch1 = []
    ch2 = []
    for gene1, gene2 in zip(p1, p2):
        # Calculate the min and max for the blending range
        lower_bound = min(gene1, gene2) - alpha * abs(gene1 - gene2)
        upper_bound = max(gene1, gene2) + alpha * abs(gene1 - gene2)
        
        # Generate two offspring by sampling from the range [lower_bound, upper_bound]
        # highly explorative character
        ch1.append(np.random.uniform(lower_bound, upper_bound))
        ch2.append(np.random.uniform(lower_bound, upper_bound))
    
    return ch1, ch2

def crossover (all_parents, p_crossover, 
               recombination_operator:callable) -> list:
    ''''Perform whatever kind of recombination and produce all offspring'''
    offspring = []
    # Make sure all parents mixed 
    np.random.shuffle(all_parents)
    for parent1, parent2 in zip(all_parents[::2], all_parents[1::2]):
        # no recombination, return parents
        if np.random.uniform() > p_crossover:
            offspring.append(parent1) # ch1
            offspring.append(parent2) # ch2
        else:
            # perform recombination
            # (possibly adapt at different stages of evolution?)
            ch1, ch2 = recombination_operator(parent1, parent2)
            offspring.append(ch1)
            offspring.append(ch2)
    return np.array(offspring).reshape(-1, len(all_parents[0]))

def uncorrelated_mut_one_sigma(individual, sigma, mutation_rate):
    ''' 
    Apply uncorrelated mutation with one step size
    tau = 1/sqrt(n), n = problem size
	SD' = SD * e**(N(0,tau))
    x'i = xi + SD' * N(0,1) 
    '''
    tau = 1/np.sqrt(len(individual))
    sigma_prime = sigma * np.exp(np.random.normal(0,tau))
    individual_mutated = []
    for xi in individual:
        if np.random.uniform() > mutation_rate:
            individual_mutated.append(xi)
        else:
            xi_prime = xi + sigma_prime * np.random.standard_normal()
            # make sure that within genetic code
            corrected_xi = within_genetic_code(xi_prime, [-1.0,1.0])
            individual_mutated.append(corrected_xi)
        
    return individual_mutated, sigma_prime

# ToDo:
def uncorrelated_mut_N_sigmas():
    pass

# ToDo: 
def save_results():
    pass

if __name__ == '__main__':
    main()
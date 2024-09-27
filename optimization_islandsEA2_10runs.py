###############################################################################
# Author: Matus Halak       			                                      #
# matus.halak@gmail.com   
# 
# In this script I implement Speciation (ISLANDS MODEL) as a way to increase diversity
# GOALS: also implement uncorrelated mutation with N sigmas!!! (put also in basic script)
# Figure out how to use numpy arrays preferably
# save results with a function

# good parameters: pop 100/120/140/150; nislands 4/5/6/7 gpi 10/20 mr .6 cr.3 ()
# python optimization_islandsEA2.py -nme 7 -pop 120 -mg 100 -name test -mr .6 -cr .3 -gpi 10 -nislands 5  
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
from pandas import read_csv

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-pop', '--popsize', type=int, required=False, default = 150, help="Population size (eg. 100)")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 100, help="Max generations (eg. 500)")
    parser.add_argument('-cr', '--crossover_rate', type=float, required=False, default = 0.85, help="Crossover rate (e.g., 0.8)")
    parser.add_argument('-mr', '--mutation_rate', type=float, required=False, default = 0.25, help="Mutation rate (e.g., 0.05)")
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default = 10, help="Number of Hidden Neurons (eg. 10)")
    parser.add_argument('-tst', '--test', type=bool, required=False, default = False, help="Train or Test (default = Train)")
    parser.add_argument('-enemies', '--enemies', nargs='+', type=int, default=[5, 6, 8], required=False, help="List of enemies (default: 5, 6, 8)")
    parser.add_argument('-dir', '--directory', type=str, default='EA2_line_plot_runs', required=False, help="Directory to save runs")
    parser.add_argument('-nislands','--nislands', type = int, required = False, default = 5, help = 'Select number of islands')
    parser.add_argument('-gpi', '--gen_per_isl', type = int, required=False, default=10, help='Generations / island')

    return parser.parse_args()

def main():
    '''Main function for basic EA, runs the islands algorithm which saves results for multiple enemies'''
    # command line arguments for experiment parameters
    args = parse_args()
    popsize = args.popsize
    mg = args.maxgen
    cr = args.crossover_rate
    mr = args.mutation_rate
    n_hidden = args.nhidden
    enemies = args.enemies
    nislands = args.nislands
    gpi = args.gen_per_isl
    base_dir = args.directory

    # Create the base directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Run the islands algorithm for each enemy
    for enemy in enemies:
        enemy_dir = os.path.join(base_dir, f'EN{enemy}')
        if not os.path.exists(enemy_dir):
            os.makedirs(enemy_dir)

        for run in range(1, 11):  # Run the algorithm 10 times for each enemy
            run_dir = os.path.join(enemy_dir, f'run_{run}_EN{enemy}')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            # Set up the environment
            env = Environment(experiment_name=run_dir,
                              enemies=[enemy],
                              playermode="ai",
                              player_controller=player_controller(n_hidden),
                              enemymode="static",
                              level=2,
                              speed="fastest",
                              visuals=False)
            
             # default environment fitness is assumed for experiment
            env.state_to_log() # checks environment state

            # Run the islands algorithm for this enemy and run number
            print(f'\nRunning islands algorithm for enemy {enemy}, run {run}\n')
            islands(popsize, mg, mr, cr, n_hidden, run_dir, env, n_islands=nislands, gen_per_island=gpi)


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

# def speciation
def make_islands(n_islands:int, popsize:int, ind_size:int):
    if popsize % n_islands != 0 or popsize <= 0:
        popsize = int(input(f'Enter population size divisible by number of islands ({n_islands}):'))
    # islands = np_array (rows = islands, columns = individuals, depth = genes)
    return np.random.uniform(-1.0,1.0,(n_islands, popsize//n_islands, ind_size))
  
def island_life(island, generations_per_island,env, mr, cr):
    # island is a 3D array - 1(island) x popsize x n_genes
    popsize, indsize = island.shape[-2:]
    pop = np.reshape(island,(popsize, indsize)) # np 2D array popsize x n_genes
    fit = evaluate_fitnesses(env,pop) # np 1d array popsize
    
    # Start tracking best individual
    best_idx = np.argmax(fit) # indexes row of pop / fitness value of best individual
    best_individual = pop[best_idx,:] # np 1D array n_genes
    best_fitness = fit[best_idx]

    all_time = (best_individual, best_fitness)
    
    sigma_prime = .05
    # Gen, Best Fit, Mean Fit, SD Fit
    gen_fits, gen_pops = [], []
    for g in range(generations_per_island):
        # niching
        shared_fit = vectorized_fitness_sharing(fit, pop, [-1.0, 1.0]) # np 1d array popsize
        # Parent selection: (Tournament? - just first try) Probability based - YES
        parents, parent_fitnesses = vectorized_parent_selection(pop, shared_fit, env) # np 2d array (popsize x n_genes) & np 1d array popsize

        # crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
        offspring = vectorized_crossover(parents, cr, vectorized_blend_recombination)
        
        # mutation: Uncorrelated mutation with N step sizes
        offspring_mutated, sigma_primes = zip(*[vectorized_uncorrelated_mut_one_sigma(offspring[ind,:], sigma_prime, mr) for ind in range(offspring.shape[0])])
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

        # get every generation & its fitnesses
        gen_fits.append(fit)
        gen_pops.append(pop)
        if best_fitness > all_time[-1]:
            all_time = (best_individual, best_fitness)
        
    np.reshape(pop, (1,popsize, indsize)) # back to island format
    return pop, fit, all_time, gen_fits, gen_pops

def vectorized_migration(islands: np.ndarray, N: int = 5) -> np.ndarray:
    """
    Exchange 1/N of the population between islands using a ring-island topology.
    """
    # Number of islands and individuals per island
    num_islands, num_individuals, _ = islands.shape
    
    # Create a shifted version of the islands array for ring topology (shift by 1)
    shifted_islands = np.roll(islands, shift=-1, axis=0)
    
    # Calculate how many individuals to exchange
    num_to_migrate = num_individuals // N
    
    # Vectorized migration: 
    # Replace the first `num_to_migrate` individuals in each island from the last `num_to_migrate` individuals of the previous island
    islands[:, :num_to_migrate, :] = shifted_islands[:, -num_to_migrate:, :]
    
    return islands

# Define a wrapper for parallelization (joblib)
def parallel_island_life(island_slice, gen_per_island,
                         name, contr, enemies, mr, cr):
    # Reinitialize or avoid non-picklable objects inside this function if needed
    env = initialize_env(name, contr, enemies)  # Re-initialize the environment inside each worker if necessary
    return island_life(island_slice, gen_per_island, env, mr, cr)

def gain_diversity(env, best_sol, population_genes):
    '''
    Obtains gain for best solution in a generation and population diversity of a given generation (average genewise STD)
    '''
    # Gain
    _, pl, el, _ = run_game(env,best_sol, test=True)
    gain = pl - el

    # Diversity: rows = individuals, columns = genes
    genewise_stds = np.std(population_genes, axis = 1) # per gene, across individuals
    diversity = np.mean(genewise_stds)

    return gain, diversity

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
    all_gen_res = []
    best, bf = [], 0
    for cycle in range(max_gen//gen_per_island):
        # Parallelize island life across islands
        results = Parallel(n_jobs=-1)(delayed(parallel_island_life)(islands[isl, :, :], gen_per_island, name, contr, enemies, mr, cr) 
                                      for isl in range(islands.shape[0]))
        
        # Unzip results into evolved islands and best individuals
        islands_evolved, fitnesses_evoled, islands_best, gen_fits, gen_pops = zip(*results)

        # gen data = 5 islands x N generations / island x Island population
        all_gen_res.append(process_gen_data(gen_fits, gen_pops, cycle))
        fitnesses_evoled = np.array(fitnesses_evoled)
        islands_evolved = np.array(islands_evolved)
        islands_best = list(islands_best)
        best_fits = [ib[-1] for ib in islands_best]
        i_best = np.argmax(best_fits)
        
        fin = time.time()
        gain, diversity = gain_diversity(env, islands_best[i_best][0] ,islands_evolved)
        print(f'Cycle {cycle} best fitness: {islands_best[i_best][-1]}, mean fitness: {np.mean(fitnesses_evoled)}, SD fitness: {np.std(fitnesses_evoled)} , DIVERSITY: {diversity} , t:{fin-ini}')
        if islands_best[i_best][-1] > bf:
            best = islands_best[i_best][0]
            bf = islands_best[i_best][-1]
            np.savetxt(experiment_name+'/alltime.txt',best)
        # migrate between islands
        np.random.shuffle(islands_evolved) # get rid of order
        islands = np.array(vectorized_migration(islands_evolved)).reshape((n_islands,-1, individual_dims))
    
    save_results(all_gen_res, experiment_name)
    return islands_best
    # run second half of generations in normal evolutionary mode (mixing everyone together)

def process_gen_data(gen_fits, gen_pops, cycle):
    '''
    Want to get data / generation across all islands
    '''
    # gen fits = n islands x N generations / island x Island population
    # eg. 5 : 10 : 20
    # max, mean std diversity
    gens_fits = np.array(gen_fits)
    gens_pops = np.array(gen_pops)
    # maxes
    gen_maxes = np.max(gens_fits, axis = 2).max(axis = 0) # 10 values
    # mean
    gen_means = np.mean(gens_fits, axis = 2).mean(axis = 0) # 10 values
    # stds
    gen_stds = np.std(gens_fits, axis = 2).std(axis = 0) # 10 values
    # Diversity (ISLANDS 4D): island : generation : individuals : gene 
    genewise_stds = np.std(gens_pops, axis = 2) # per gene, across individuals
    diversity = np.mean(genewise_stds, axis = 2).mean(axis = 0) # 10 values

    rows = np.arange(round(0+diversity.shape[0]*cycle), 
                     round(diversity.shape[0]+diversity.shape[0]*cycle))

    return np.vstack([rows, gen_maxes, gen_means, gen_stds, diversity]).T

# promoting diversity
def vectorized_fitness_sharing(fitnesses: np.ndarray, population: np.ndarray, 
                               gene_limits: tuple[float, float], k=0.15) -> np.ndarray:
    """
    Apply fitness sharing as described in the slide to promote diversity and niche creation.
    [vectorized for efficiency]
    """
    # Calculate the max possible distance between two genes
    gene_distance_max = gene_limits[1] - gene_limits[0]  
    
    # Calculate max possible distance between two individuals and sigma_share
    max_possible_distance = gene_distance_max * np.sqrt(population.shape[1]) 
    # radius which we consider 'close' <- HYPERPARAMETER!
    sigma_share = k * max_possible_distance  

    # Calculate pairwise Euclidean distances for the entire population
    diff_matrix = population[:, np.newaxis] - population[np.newaxis, :]
    distances = np.linalg.norm(diff_matrix, axis=2)  # Shape: (pop_size, pop_size)

    # Apply the niche count function where distance < sigma_share
    niche_matrix = np.where(distances < sigma_share, 1 - (distances / sigma_share), 0)

    # Each individual includes its own contribution (sh(d) = 1 for distance 0 with itself)
    np.fill_diagonal(niche_matrix, 1)
    
    # Calculate the niche count for each individual
    niche_counts = niche_matrix.sum(axis=1)

    # Calculate shared fitnesses
    shared_fitnesses = fitnesses / niche_counts

    if all(shared_fitnesses < 100) == False:
        breakpoint()
    return shared_fitnesses

# Tournament selection
def vectorized_tournament_selection(population, fitnesses, n_tournaments, k=2):
    """
    Vectorized tournament selection to select multiple parents in parallel.
    !!! MUCH MORE EFFICIENT!!!
    k SHOULD be ± 10% pop size
    """
    # Randomly select k individuals for each tournament (by default with replacement)
    players = np.random.choice(np.arange(len(population)), size=(n_tournaments, k))

    # Find the best individual (highest fitness) in each tournament
    best_indices = np.argmax(fitnesses[players], axis=1)

    # Retrieve the winning individuals (parents) from the population
    selected_parents = population[players[np.arange(n_tournaments), best_indices]]
    
    return selected_parents

# Parent selection
def vectorized_parent_selection(population, fitnesses, env: Environment, n_children=2, k=15):
    """
    Vectorized parent selection using tournament selection.
    """
    n_parents = int(len(population) / n_children) * n_children  # Ensure multiple of n_children
    
    # Perform tournament selection for all parents at once
    g_parents = vectorized_tournament_selection(population, fitnesses, n_parents, k)
    
    # Vectorized fitness evaluation of selected parents
    f_parents = evaluate_fitnesses(env, g_parents)
    return g_parents, f_parents

# Survivor selection with elitism and random diversity preservation
def survivor_selection(parents, parent_fitnesses, 
                       offspring, offspring_fitnesses, elite_fraction=0.8):
    """
    Select survivors using elitism with some randomness to maintain diversity.
    Fixed a MAJOR MISTAKE HERE! 
    (had to do with adding numpy arrays - X CONCATENATE!!!)
    """
    # Combine parents and offspring
    total_population = np.vstack([parents, offspring])
    total_fitnesses = np.hstack([parent_fitnesses, offspring_fitnesses])
    # Sort by fitness in descending order
    # Sort the indices based on fitness in descending order
    sorted_indices = np.argsort(total_fitnesses)[::-1]  # argsort returns indices, ::-1 reverses the order

    # Apply the sorted indices to sort both fitnesses and the population
    sorted_population = total_population[sorted_indices,:]
    sorted_fitnesses = total_fitnesses[sorted_indices]

    # Select elites (the top portion of the population)
    num_elites = int(elite_fraction * parents.shape[0])  # Determine number of elites to preserve
    elites = sorted_population[:num_elites,:]
    efs = sorted_fitnesses[:num_elites]

    # Select the remaining individuals randomly from the rest to maintain diversity
    remaining_individuals = sorted_population[num_elites:,:]
    remaining_fs = sorted_fitnesses[num_elites:]
    # Shuffle INDICES to add randomness (need to do indices 
    # to keep relationship between individuals:fitnesses same)
    shuffled_indices = np.random.permutation(remaining_fs.shape[0])  
    # Select the remaining individuals to fill the population
    num_remaining = parents.shape[0] - num_elites
    remaining_indices = shuffled_indices[:num_remaining]
    # remaining individuals
    selected_remaining = remaining_individuals[remaining_indices,:]
    # remaining fitnesses
    selected_remaining_fs = remaining_fs[remaining_indices]
    
    # Combine elites and randomly selected individuals
    survivors = np.vstack([elites, selected_remaining])
    survivor_fs = np.hstack([efs, selected_remaining_fs])
    # Separate fitnesses and individuals for the return
    return survivors, survivor_fs

def vectorized_blend_recombination(p1: np.ndarray, p2: np.ndarray, 
                                   alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    '''Apply blend recombination (BLX-alpha) to create two offspring.
    --> really explorative [vectorized for efficiency]'''
    # Calculate the lower and upper bounds for the blending range
    lower_bound = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
    upper_bound = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)

    # Generate two offspring by sampling from the range [lower_bound, upper_bound]
    ch1 = np.random.uniform(lower_bound, upper_bound)
    ch2 = np.random.uniform(lower_bound, upper_bound)

    return ch1, ch2

def vectorized_crossover(all_parents: np.ndarray, p_crossover: float, 
                         recombination_operator: callable) -> np.ndarray:
    ''''
    Perform whatever kind of recombination and produce all offspring 
    [vectorized for efficiency]
    '''
    # Shuffle the parents to ensure random pairs
    np.random.shuffle(all_parents)
    
    # Split the parents into pairs
    half_pop_size = len(all_parents) // 2
    parent1 = all_parents[:half_pop_size,:]
    parent2 = all_parents[half_pop_size:,:]

    # Decide which pairs will crossover based on the crossover probability
    crossover_mask = np.random.uniform(0, 1, size=half_pop_size) < p_crossover

    # Perform recombination on selected pairs
    ch1, ch2 = recombination_operator(parent1, parent2)

    # Create offspring array by filling in the crossover children and copying the parents when no crossover
    offspring = np.empty_like(all_parents)

    # Assign offspring from crossover or retain parents
    offspring[:half_pop_size] = np.where(crossover_mask[:, np.newaxis], ch1, parent1)
    offspring[half_pop_size:2 * half_pop_size] = np.where(crossover_mask[:, np.newaxis], ch2, parent2)

    return offspring

def vectorized_uncorrelated_mut_one_sigma(individual: np.ndarray, sigma: float, 
                                          mutation_rate: float) -> tuple[np.ndarray, float]:
    ''' 
    Apply uncorrelated mutation with one step size [vectorized for efficiency]
    tau = 1/sqrt(n), n = problem size
	SD' = SD * e**(N(0,tau))
    x'i = xi + SD' * N(0,1) 
    '''
    # Calculate the tau parameter
    tau = 1 / np.sqrt(len(individual))
    
    # Calculate the new mutation step size (sigma_prime)
    sigma_prime = sigma * np.exp(np.random.normal(0, tau))

    # Create a mutation mask (True where mutation will be applied, False otherwise)
    mutation_mask = np.random.uniform(0, 1, size=len(individual)) < mutation_rate

    # Apply the mutation only where the mask is True
    mutations = np.random.standard_normal(size=len(individual)) * sigma_prime
    mutated_individual = np.where(mutation_mask, individual + mutations, individual)

    # Correct the mutated genes to be within the genetic code bounds [-1.0, 1.0]
    mutated_individual = np.clip(mutated_individual, -1.0, 1.0)

    return mutated_individual, sigma_prime

# ToDo: 
def save_results(res, experiment_name):
        res = np.vstack(res)
        np.savetxt(experiment_name+'/results.txt',res, header="gen best mean std diversity", fmt = '%.6f')

if __name__ == '__main__':
    main()
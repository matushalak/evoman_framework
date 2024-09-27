###############################################################################
# Author: Matus Halak       			                                      #
# matus.halak@gmail.com     				                                  #
###############################################################################

# imports framework
import sys
import itertools
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import argparse
from joblib import Parallel, delayed
from pandas import read_csv
from threading import Lock

import random


file_lock = None

def parse_args():
    '''' Function enabling command-line arguments'''
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default=5, help="Max generations (eg. 500)")
    parser.add_argument('-nh', '--nhidden', type=int, required=False, default=10, help="Number of Hidden Neurons (eg. 10)")
    parser.add_argument('-nme', '--enemy', type=int, required=False, default=6, help="Select Enemy")
    return parser.parse_args()


def mean_result_EA1(hyperparameters, popsize, mg, n_hidden, experiment_name, env, new_evolution, save_gens, num_reps):
    avg_fitness = 0
    for _ in range(num_reps):
        fitness = basic_ea(hyperparameters, popsize, mg, n_hidden, experiment_name, env, new_evolution, save_gens)
        avg_fitness += fitness
    return avg_fitness / num_reps


def initialize_lock():
    global file_lock
    if file_lock is None:
        file_lock = Lock()  # Initialize the lock for each worker

def save_results(experiment_name, params, fitness):
    """ Save the fitness and corresponding parameters to a text file with a lock """
    global file_lock
    initialize_lock()  # Ensure the lock is initialized in each worker process
    file_path = os.path.join(experiment_name, 'grid_search_results.txt')

    # Use the file lock to ensure that only one process writes to the file at a time
    with file_lock:
        with open(file_path, 'a') as f:
            f.write(f"Params: Mutation Rate={params['mutation_rate']}, Crossover Rate={params['crossover_rate']}, "
                    f"Population Size={params['popsize']}\n")
            f.write(f"Fitness: {fitness}\n\n")

def aggregate_results(experiment_name):
    """ Aggregates the final results and prints the best fitness with parameters """
    final_file = os.path.join(experiment_name, 'grid_search_results.txt')

    best_fitness = -float('inf')  # Initialize with a very low value
    best_params = None

    # Read the final results file and find the best fitness
    with open(final_file, 'r') as infile:
        lines = infile.readlines()

        # Read two lines at a time (params and fitness)
        for i in range(0, len(lines), 2):
            params_line = lines[i]
            fitness_line = lines[i + 1]

            # Extract the fitness value
            fitness_value = float(fitness_line.split(":")[1].strip())

            # Check if this is the best fitness so far
            if fitness_value > best_fitness:
                best_fitness = fitness_value
                best_params = params_line

    # Print the best fitness and parameters
    print("\nBest Fitness:", best_fitness)
    print("Best Parameters:", best_params)


def evaluate_combination(mutation_rate, crossover_rate, popsize, mg, n_hidden, experiment_name, enemy, new_evolution, save_gens, num_reps):
    # Create a new environment for each worker (avoid passing the env object)
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    env.state_to_log()

    # Set the hyperparameters
    scaling_factor = 0.15
    sigma_prime = 0.05
    alpha = 0.5
    tournament_size = 15
    elite_fraction = 0.8

    hyperparameters = (scaling_factor, sigma_prime, alpha, tournament_size, elite_fraction, mutation_rate, crossover_rate)

    # Evaluate performance using mean_result_EA1
    fitness = mean_result_EA1(hyperparameters, popsize, mg, n_hidden, experiment_name, env, new_evolution, save_gens, num_reps)

    # Save results to a file
    save_results(experiment_name, {'mutation_rate': mutation_rate, 'crossover_rate': crossover_rate, 'popsize': popsize}, fitness)

    return fitness


def main():
    '''Main function for grid search EA'''
    args = parse_args()
    mg = args.maxgen
    n_hidden = args.nhidden
    enemy = args.enemy

    save_gens = False
    num_reps = 1
    new_evolution = True

    experiment_name = 'basic_' + (args.exp_name if isinstance(args.exp_name, str) else input("Enter Experiment (directory) Name:"))
    experiment_name += f'_{enemy}'

    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Define grid search ranges for mutation rate, crossover rate, and population size
    mutation_rates = [0.05, 0.25]#, 0.45, 0.65, 0.85]
    crossover_rates = [0.05, 0.25]#, 0.45, 0.65, 0.85]
    population_sizes = [100]#, 125, 150, 175, 200]

    # Cartesian product of the grid search parameters
    grid = list(itertools.product(mutation_rates, crossover_rates, population_sizes))

    # Run the grid search in parallel using joblib.Parallel
    Parallel(n_jobs=-1)(delayed(evaluate_combination)(mutation_rate, crossover_rate, popsize, mg, n_hidden, experiment_name,
                                                     enemy, new_evolution, save_gens, num_reps)
                        for mutation_rate, crossover_rate, popsize in grid)




# for parallelization later
# worker_env = None




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
    fitnesses = Parallel(n_jobs=-1)(
        delayed(run_game_in_worker)(name, contr, enemies, ind) for ind in population
    )
    return np.array(fitnesses)

def run_game_in_worker(name, contr, enemies, ind):
    # Recreate or reinitialize the environment from env_config inside the worker
    # global worker_env
    # if worker_env is None:
    worker_env = Environment(experiment_name=name,
                            enemies=enemies,
                            playermode="ai",
                            player_controller=contr, # you  can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
    return run_game(worker_env, ind)

# initializes random population of candidate solutions
def initialize_population(popsize, individual_dims, gene_limits:list[float, float]):
    ''' Generate a population of 
        N = popsize solutions, each solution containing
        N = individual_dims genes, each gene within <gene_limits>'''
    population = np.random.uniform(*gene_limits, (popsize, individual_dims))
    return population

# promoting diversity
def vectorized_fitness_sharing(fitnesses: np.ndarray, population: np.ndarray, 
                               gene_limits: tuple[float, float], k=0.15) -> np.ndarray:
    """
    Apply fitness sharing as described in the slide to promote diversity and niche creation.
    [vectorized for efficiency]
    
    Parameters:
    fitnesses (np.ndarray): Fitness values of the population.
    population (np.ndarray): Population of individuals as a 2D numpy array.
    gene_limits (tuple): Minimum and maximum gene values as (min, max).
    k (float): Scaling factor controlling fitness sharing radius.
    
    Returns:
    np.ndarray: Shared fitness values.
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
def vectorized_tournament_selection(population, fitnesses, n_tournaments, k=15):
    """
    Vectorized tournament selection to select multiple parents in parallel.
    
    Parameters:
    population (ndarray): Array of individuals in the population.
    fitnesses (ndarray): Array of fitnesses corresponding to the population.
    n_tournaments (int): Number of tournaments (i.e., number of parents to select).
    k (int): Tournament size (number of individuals in each tournament).
    
    Returns:
    ndarray: Selected parent genotypes.
    """
    # Randomly select k individuals for each tournament (by default with replacement)
    players = np.random.choice(np.arange(len(population)), size=(n_tournaments, k))

    # Find the best individual (highest fitness) in each tournament
    best_indices = np.argmax(fitnesses[players], axis=1)

    # Retrieve the winning individuals (parents) from the population
    selected_parents = population[players[np.arange(n_tournaments), best_indices]]
    
    return selected_parents

# Parent selection
def vectorized_parent_selection(population, fitnesses, env: Environment, k=15, n_children=2):
    """
    Vectorized parent selection using tournament-based selection.
    
    Parameters:
    population (ndarray): The population of individuals.
    fitnesses (ndarray): The corresponding fitnesses of individuals.
    env (Environment): The simulation environment for fitness evaluation.
    n_children (int): Number of children per pair of parents.
    k (int): Tournament size for selection.
    
    Returns:
    g_parents (ndarray): Selected parent genotypes.
    f_parents (ndarray): Fitnesses of selected parents.
    """
    n_parents = int(len(population) / n_children) * n_children  # Ensure multiple of n_children
    
    # Perform tournament selection for all parents at once
    g_parents = vectorized_tournament_selection(population, fitnesses, n_parents, k)
    
    # Vectorized fitness evaluation of selected parents
    f_parents = evaluate_fitnesses(env, g_parents)
    # breakpoint()
    return g_parents, f_parents

# Survivor selection with elitism and random diversity preservation
def survivor_selection(parents, parent_fitnesses, 
                       offspring, offspring_fitnesses, elite_fraction=0.8):
    """Select survivors using elitism with some randomness to maintain diversity."""
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

def vectorized_blend_recombination(p1: np.ndarray, p2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized blend recombination (BLX-alpha) to create two offspring.
    
    Parameters:
    p1 (np.ndarray): Parent 1 as an array.
    p2 (np.ndarray): Parent 2 as an array.
    alpha (float): Alpha parameter for the blending range.
    
    Returns:
    tuple: Two offspring arrays (ch1, ch2).
    """
    # Calculate the lower and upper bounds for the blending range
    lower_bound = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
    upper_bound = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)

    # Generate two offspring by sampling from the range [lower_bound, upper_bound]
    ch1 = np.random.uniform(lower_bound, upper_bound)
    ch2 = np.random.uniform(lower_bound, upper_bound)

    return ch1, ch2


def vectorized_crossover(all_parents: np.ndarray, p_crossover: float, alpha:float, recombination_operator: callable) -> np.ndarray:
    """
    Perform fully vectorized recombination to produce all offspring.
    """
    # Shuffle the parents to ensure random pairs
    np.random.shuffle(all_parents)
    
    # Split the parents into pairs
    half_pop_size = len(all_parents) // 2
    parent1 = all_parents[:half_pop_size,:]
    parent2 = all_parents[half_pop_size:,:]

    # Decide which pairs will crossover based on the crossover probability
    crossover_mask = np.random.uniform(0, 1, size=half_pop_size) < p_crossover

    # Perform recombination on selected pairs
    ch1, ch2 = recombination_operator(parent1, parent2, alpha)

    # Create offspring array by filling in the crossover children and copying the parents when no crossover
    offspring = np.empty_like(all_parents)

    # Assign offspring from crossover or retain parents
    offspring[:half_pop_size] = np.where(crossover_mask[:, np.newaxis], ch1, parent1)
    offspring[half_pop_size:2 * half_pop_size] = np.where(crossover_mask[:, np.newaxis], ch2, parent2)

    return offspring

def vectorized_uncorrelated_mut_one_sigma(individual: np.ndarray, sigma: float, mutation_rate: float) -> tuple[np.ndarray, float]:
    """
    Apply uncorrelated mutation with one step size [vectorized for efficiency]
    tau = 1/sqrt(n), n = problem size
	SD' = SD * e**(N(0,tau))
    x'i = xi + SD' * N(0,1) 
    """
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

def basic_ea (hyperparameters:tuple, popsize:int, max_gen:int, n_hidden_neurons:int,
              experiment_name:str, env:Environment, new_evolution:bool = True, save_gens:bool = True):
    ''' 
    Basic evolutionary algorithm to optimize the weights 
    '''
    random_number = random.randint(1, 500)

    scaling_factor, sigma_prime, alpha, tournament_size, elite_fraction, mr, cr = hyperparameters   

    # number of weights for multilayer with 10 hidden neurons
    individual_dims = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    # initiation time
    ini = time.time()

    # Initialization for new experiment
    if new_evolution == True:
        # Initialize population & calculate fitness
        ini_g = 0
        gene_limits = [-1.0, 1.0]
        # starting step size 0.5?
        #sigma_prime = 0.05     --> NOW USE AS HYPERPARAMETER FOR OPTIMIZATION
        population = initialize_population(popsize, individual_dims, gene_limits)  
        fitnesses = evaluate_fitnesses(env, population)
        solutions = [population, fitnesses]
        env.update_solutions(solutions)

        # stats
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        # Start tracking best individual
        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]

        all_time = best_individual, best_fitness

        # stagnation prevention
        stagnation = 0
        starting_mutation_rate, starting_crossover_rate = mr, cr

        # saves results for first pop
        if save_gens == True:
            file_aux  = open(experiment_name+'/results.txt','a')
            file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')
            print( '\n GENERATION '+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '
                +str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
            file_aux.write('\n'+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '+str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
            file_aux.close()

    # evolution loop
    for i in range(max_gen):
        # niching (fitness sharing)
        shared_fitnesses = vectorized_fitness_sharing(fitnesses, population, gene_limits, scaling_factor)
        # shared_fitnesses = fitnesses # disables fitness sharing
        # ?? crowding ??
        # ?? speciation - islands ??

        # Parent selection: (Tournament? - just first try) Probability based - YES
        parents, parent_fitnesses = vectorized_parent_selection(population, shared_fitnesses, env, tournament_size)

        # crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
        offspring = vectorized_crossover(parents, cr, alpha, vectorized_blend_recombination)
        # offspring_fitnesses = evaluate_fitnesses(env, offspring)
        
        # mutation: Uncorrelated mutation with N step sizes
        offspring_mutated, sigma_primes = zip(*[vectorized_uncorrelated_mut_one_sigma(ind, sigma_prime, mr) for ind in offspring])
        offspring_fitnesses = evaluate_fitnesses(env, offspring_mutated)
        if all(offspring_fitnesses < 100) == False:
            breakpoint()

        sigma_prime = sigma_primes[np.argmax(offspring_fitnesses)]

        # Survivor selection with elitism & some randomness
        population, fitnesses = survivor_selection(parents, parent_fitnesses, 
                                                   list(offspring_mutated), offspring_fitnesses, elite_fraction)

        # Check for best solution
        best_idx = np.argmax(fitnesses)
        generational_fitness = fitnesses[best_idx]
        gen_mean = np.mean(fitnesses)
        gen_std = np.std(fitnesses)
        
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]

        if best_fitness > all_time[-1]:
            all_time = (best_individual, best_fitness)

        # OUTPUT: weights + biases vector
        # saves results
        if save_gens == True:
            file_aux  = open(experiment_name+'/results.txt','a')
            # file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')
            file_aux.write('\n'+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '+str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
            file_aux.close()

            # saves generation number
            file_aux  = open(experiment_name+'/gen.txt','a')
            file_aux.write(str(i))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(experiment_name+'/best.txt',best_individual)
            np.savetxt(experiment_name+'/alltime.txt',all_time[0])

            if not os.path.exists('basic_solutions'):
                os.makedirs('basic_solutions')
            np.savetxt(f'basic_solutions/{env.enemyn}best.txt', best_individual)

        #print( '\n GENERATION '+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '
        #      +str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
        print(f'\n{random_number}, GENERATION '+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '
              +str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))


        # saves simulation state
        solutions = [population, fitnesses]
        env.update_solutions(solutions)
        env.save_state()
        
    if save_gens == True:
        np.savetxt(experiment_name+'/alltime.txt',all_time[0])
        np.savetxt(f'basic_solutions/{env.enemyn}alltime.txt', all_time[0])

        file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
        file.close()

    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')

    env.state_to_log() # checks environment state

    return all_time[-1]



if __name__ == '__main__':
    main()
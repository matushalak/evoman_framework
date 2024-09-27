###############################################################################
# Author: Matus Halak       			                                      #
# matus.halak@gmail.com     				                                  #
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
    parser.add_argument('-enemies', '--enemies', nargs='+', type=int, default=[5, 6, 8], required=False, help="List of enemies (e.g., 2 6 8)")
    parser.add_argument('-dir', '--directory', type=str, default='EA1_line_plot_runs', required=False, help="Directory to save runs")
    #EA1_line_plot_runs

    return parser.parse_args()


def main():
    '''Main function for basic EA, runs the EA for multiple enemies'''
    # command line arguments for experiment parameters
    args = parse_args()
    popsize = args.popsize
    mg = args.maxgen
    cr = args.crossover_rate
    mr = args.mutation_rate
    n_hidden = args.nhidden
    enemies = args.enemies
    base_dir = args.directory

    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Create the base directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Run the EA for each enemy
    for enemy in enemies:
        enemy_dir = os.path.join(base_dir, f'EN{enemy}')
        if not os.path.exists(enemy_dir):
            os.makedirs(enemy_dir)

        for run in range(1, 11):  # Run the EA 10 times for each enemy
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

            # Run the basic EA for this enemy and run number
            print(f'\nRunning EA for enemy {enemy}, run {run}\n')
            basic_ea(popsize, mg, mr, cr, n_hidden, run_dir, env)

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
def vectorized_parent_selection(population, fitnesses, env: Environment, n_children=2, k=15):
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


def vectorized_crossover(all_parents: np.ndarray, p_crossover: float, recombination_operator: callable) -> np.ndarray:
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
    ch1, ch2 = recombination_operator(parent1, parent2)

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

def gain_diversity(env, best_sol, population_genes):
    '''
    Obtains gain for best solution in a generation and population diversity of a given generation (average genewise STD)
    '''
    # Gain
    _, pl, el, _ = run_game(env,best_sol, test=True)
    gain = pl - el

    # Diversity: rows = individuals, columns = genes
    genewise_stds = np.std(population_genes, axis = 0) # per gene, across individuals
    diversity = np.mean(genewise_stds)

    return gain, diversity


def basic_ea (popsize:int, max_gen:int, mr:float, cr:float, n_hidden_neurons:int,
              experiment_name:str, env:Environment, new_evolution:bool = True):
    ''' 
    Basic evolutionary algorithm to optimize the weights 
    '''
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
        sigma_prime = 0.05
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

        # for self-adaptivity
        starting_mutation_rate, starting_crossover_rate = mr, cr
        elite_fraction, starting_elite_fraction = 0.8, 0.8

        gain, diversity = gain_diversity(env, best_individual, population)
        # saves results for first pop
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std gain diversity')
        print( '\n GENERATION '+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '
              +str(round(gain, 6))+' '+str(round(diversity, 6)))
        file_aux.write('\n'+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '+str(round(gain, 6))+' '+str(round(diversity, 6)))
        file_aux.close()

    # evolution loop
    for i in range(max_gen):
        # niching (fitness sharing)
        shared_fitnesses = vectorized_fitness_sharing(fitnesses, population, gene_limits)
        # shared_fitnesses = fitnesses # disables fitness sharing
        # ?? crowding ??
        # ?? speciation - islands ??

        # Parent selection: (Tournament? - just first try) Probability based - YES
        parents, parent_fitnesses = vectorized_parent_selection(population, shared_fitnesses, env)

        # crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
        offspring = vectorized_crossover(parents, cr, vectorized_blend_recombination)
        # offspring_fitnesses = evaluate_fitnesses(env, offspring)
        
        # mutation: Uncorrelated mutation with N step sizes
        offspring_mutated, sigma_primes = zip(*[vectorized_uncorrelated_mut_one_sigma(ind, sigma_prime, mr) for ind in offspring])
        offspring_fitnesses = evaluate_fitnesses(env, offspring_mutated)
        if all(offspring_fitnesses < 100) == False:
            breakpoint()

        sigma_prime = sigma_primes[np.argmax(offspring_fitnesses)]

        # Survivor selection with elitism & some randomness
        population, fitnesses = survivor_selection(parents, parent_fitnesses, 
                                                   list(offspring_mutated), offspring_fitnesses,elite_fraction)

        # Check for best solution
        best_idx = np.argmax(fitnesses)
        generational_fitness = fitnesses[best_idx]
        gen_mean = np.mean(fitnesses)
        gen_std = np.std(fitnesses)
        
        # self adaptive to promote exploitation, TOO much exploration atm
        # if generational_fitness > 80.0:
        #     mr = .01
        #     cr = .2
        # elif generational_fitness > 85:
        #     mr = .0075
        #     cr = .175
        #     elite_fraction = .85
        # elif generational_fitness > 90:
        #     mr = .005
        #     cr = .1
        # elif generational_fitness > 92:
        #     mr = .0025
        #     cr = .05
        #     elite_fraction = .9
        # else:
        #     mr, cr = starting_mutation_rate, starting_crossover_rate 
        #     elite_fraction = starting_elite_fraction

        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]

        # stats
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        # update gain & diversity
        gain, diversity = gain_diversity(env, best_individual, population)        
    
        if best_fitness > all_time[-1]:
            all_time = (best_individual, best_fitness)

        # OUTPUT: weights + biases vector
        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        # file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')
        print( '\n GENERATION '+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '
              +str(round(gain, 6))+' '+str(round(diversity, 6)))
        file_aux.write('\n'+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '+str(round(gain, 6))+' '+str(round(diversity, 6)))
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

        # saves simulation state
        solutions = [population, fitnesses]
        env.update_solutions(solutions)
        env.save_state()

    np.savetxt(experiment_name+'/alltime.txt',all_time[0])
    np.savetxt(f'basic_solutions/{env.enemyn}alltime.txt', all_time[0])
    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log() # checks environment state

if __name__ == '__main__':
    main()
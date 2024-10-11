###############################################################################
# Authors: Matus Halak, Rick Geling, Rens Koppelman, Isis van Loenen
# matus.halak@gmail.com, rickgeling@gmail.com   
###############################################################################

# imports framework
import sys
import os

# imports other libs
import time
import numpy as np
import os
import argparse
from joblib import Parallel, delayed
from pandas import read_csv
from scipy.stats import qmc # only for initialize_population_v2

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to PYTHONPATH
sys.path.append(project_root)

from evoman.environment import Environment
from demo_controller_foldering import player_controller


# # runs game (evaluate fitness for 1 individual)
def run_game(env:Environment,individual, test=False):
    '''Runs game and returns individual solution's fitness'''
    fitfunc = 'old' #--> NOTE: change
    # vfitness, vplayerlife, venemylife, vtime
    if isinstance(individual, float):
        breakpoint()
    fitness ,p,e,t = env.play(pcont=individual)
    if test == False:
        if fitfunc == 'new':
            return (p-(2*e)) - 0.01*t
        else: 
            return fitness
    else:
        return fitness ,p,e,t
    
def run_game_in_worker(name, contr, enemies, ind):
    # Recreate or reinitialize the environment from env_config inside the worker
    # global worker_env
    # if worker_env is None:
    worker_env = Environment(experiment_name=name,
                            enemies=enemies,
                            multiplemode='yes',
                            playermode="ai",
                            player_controller=contr, # you  can insert your own controller here
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
    
    return run_game(worker_env, ind)

class ClassicEA:  #CHANGED
    def __init__(self, popsize:int, max_gen:int, mr:float, cr:float, n_hidden_neurons:int, 
                 experiment_name:str, env:Environment):  #CHANGED
        '''Initialize the EA1 class with the necessary parameters'''  #CHANGED
        self.popsize = popsize  #CHANGED
        self.max_gen = max_gen  #CHANGED
        self.mr = mr  #CHANGED
        self.cr = cr  #CHANGED
        self.n_hidden_neurons = n_hidden_neurons  #CHANGED
        self.experiment_name = experiment_name  #CHANGED
        self.env = env  #CHANGED

        # number of weights for multilayer with n_hidden_neurons (controller structure)
        self.individual_dims = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5  #CHANGED
        self.gene_limits = [-1.0, 1.0]  #CHANGED
        self.sigma_prime = 0.05  #CHANGED
        self.elite_fraction = 0.8
        self.best_individual = None  #ADDED
        self.best_fitness = None  #ADDED


    def run_evolution(self):  #CHANGED
        ''' 
        Basic evolutionary algorithm to optimize the weights 
        '''

        ini = time.time()

        # Initialization for new experiment
        ini_g = 0  #CHANGED
        population = self.initialize_population()  #CHANGED
        fitnesses = self.evaluate_fitnesses(population)  #CHANGED
        solutions = [population, fitnesses]  #ADDED
        self.env.update_solutions(solutions)  #ADDED

        # stats
        mean_fitness = np.mean(fitnesses)  #CHANGED
        std_fitness = np.std(fitnesses)  #CHANGED

        # Start tracking best individual
        best_idx = np.argmax(fitnesses)  #CHANGED
        self.best_individual = population[best_idx]  #CHANGED
        self.best_fitness = fitnesses[best_idx]  #CHANGED

        all_time = (self.best_individual, self.best_fitness)  #CHANGED

        gain, diversity = self.gain_diversity(self.env, self.best_individual, population) 

        # saves results for first population
        file_aux  = open(self.experiment_name + '/results.txt', 'a')  #CHANGED
        file_aux.write('\n\ngen best mean std gain diversity')  #CHANGED
        print('\n GENERATION ' + str(ini_g) + ' ' + str(round(self.best_fitness, 6)) + ' ' + str(round(mean_fitness, 6)) + ' ' 
            + str(round(std_fitness, 6)) + ' ' + str(round(gain, 6)) + ' ' + str(round(diversity, 6)))  #CHANGED
        file_aux.write('\n' + str(ini_g) + ' ' + str(round(self.best_fitness, 6)) + ' ' + str(round(mean_fitness, 6)) + ' ' 
                    + str(round(std_fitness, 6)) + ' ' + str(round(gain, 6)) + ' ' + str(round(diversity, 6)))  #CHANGED
        file_aux.close()  #CHANGED

        # evolution loop
        for i in range(self.max_gen):  #CHANGED

            # Niching (fitness sharing)
            shared_fitnesses = self.vectorized_fitness_sharing(fitnesses, population)  #CHANGED

            # Parent selection
            parents, parent_fitnesses = self.vectorized_parent_selection(population, shared_fitnesses)  #CHANGED

            # Crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
            offspring = self.vectorized_crossover(parents, self.cr, self.vectorized_blend_recombination)  #CHANGED

            # Mutation
            offspring_mutated, sigma_primes = zip(*[self.vectorized_uncorrelated_mut_one_sigma(ind, self.sigma_prime, self.mr) 
                                                    for ind in offspring])  #CHANGED

            offspring_fitnesses = self.evaluate_fitnesses(offspring_mutated)  #CHANGED
            if all(offspring_fitnesses < 100) == False:
                breakpoint()
            
            self.sigma_prime = sigma_primes[np.argmax(offspring_fitnesses)]  #CHANGED

            # Survivor selection
            population, fitnesses = self.survivor_selection(parents, parent_fitnesses, list(offspring_mutated), 
                                                            offspring_fitnesses, elite_fraction=self.elite_fraction)  #CHANGED

            # Check for best solution  #CHANGED
            best_idx = np.argmax(fitnesses)  #CHANGED
            generational_fitness = fitnesses[best_idx]  #CHANGED
            gen_mean = np.mean(fitnesses)  #CHANGED
            gen_std = np.std(fitnesses)  #CHANGED

            best_fitness = fitnesses[best_idx]  #CHANGED
            best_individual = population[best_idx]  #CHANGED

            # stats  #CHANGED
            mean_fitness = np.mean(fitnesses)  #CHANGED
            std_fitness = np.std(fitnesses)  #CHANGED

            # update gain & diversity  #CHANGED
            gain, diversity = self.gain_diversity(self.env, best_individual, population)  #CHANGED

            if best_fitness > all_time[-1]:  #CHANGED
                all_time = (best_individual, best_fitness)  #CHANGED

            # OUTPUT: weights + biases vector  #CHANGED
            # saves results  #CHANGED
            file_aux = open(self.experiment_name + '/results.txt', 'a')  #CHANGED
            # file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')  #CHANGED
            print('\n GENERATION ' + str(i) + ' ' + str(round(generational_fitness, 6)) + ' ' + str(round(gen_mean, 6)) + ' ' 
                + str(round(gen_std, 6)) + ' ' + str(round(gain, 6)) + ' ' + str(round(diversity, 6)))  #CHANGED
            file_aux.write('\n' + str(i) + ' ' + str(round(generational_fitness, 6)) + ' ' + str(round(gen_mean, 6)) + ' ' 
                        + str(round(gen_std, 6)) + ' ' + str(round(gain, 6)) + ' ' + str(round(diversity, 6)))  #CHANGED
            file_aux.close()  #CHANGED

            # saves generation number  #CHANGED
            file_aux = open(self.experiment_name + '/gen.txt', 'a')  #CHANGED
            file_aux.write(str(i))  #CHANGED
            file_aux.close()  #CHANGED

            # saves file with the best solution  #CHANGED
            np.savetxt(self.experiment_name + '/best.txt', best_individual)  #CHANGED
            np.savetxt(self.experiment_name + '/alltime.txt', all_time[0])  #CHANGED

            if not os.path.exists('basic_solutions'):  #CHANGED
                os.makedirs('basic_solutions')  #CHANGED
            np.savetxt(f'basic_solutions/{self.env.enemyn}best.txt', best_individual)  #CHANGED

            # saves simulation state  #CHANGED
            solutions = [population, fitnesses]  #CHANGED
            self.env.update_solutions(solutions)  #CHANGED
            self.env.save_state()  #CHANGED
        
        np.savetxt(self.experiment_name + '/alltime.txt', all_time[0])  #CHANGED
        np.savetxt(f'basic_solutions/{self.env.enemyn}alltime.txt', all_time[0])  #CHANGED
        fim = time.time()  # prints total execution time for experiment  #CHANGED
        print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')  #CHANGED
        print('\nExecution time: ' + str(round((fim - ini))) + ' seconds \n')  #CHANGED

        file = open(self.experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file  #CHANGED
        file.close()  #CHANGED

        self.env.state_to_log()  # checks environment state  #CHANGED

        print(f'All best fitness: {all_time[-1]}')  #CHANGED

        return all_time[-1]  #CHANGED

#TODO: change stuff that is a global variable: fitfunc
#TODO: define multi in run_game_in_worker


    def evaluate_fitnesses(self, population):  #CHANGED
        ''' Evaluates fitness of each individual in the population of solutions parallelized for efficiency'''  # Keeping your comment unchanged
        # Instead of passing the full environment, pass only the configuration or parameters needed to reinitialize it  # Keeping your comment unchanged
        name = self.env.experiment_name  #CHANGED
        contr = self.env.player_controller  #CHANGED
        enemies = self.env.enemies  #CHANGED
        fitnesses = Parallel(n_jobs=-1)(
            delayed(run_game_in_worker)(name, contr, enemies, ind) for ind in population  #CHANGED
        )
        return np.array(fitnesses)  # Unchanged



    def initialize_population(self):  #CHANGED
        ''' Generate a population of
            N = popsize solutions, each solution containing
            N = individual_dims genes, each gene within <gene_limits>'''  # Keeping your comment unchanged
        
        population = np.random.uniform(self.gene_limits[0], self.gene_limits[1], 
                                       (self.popsize, self.individual_dims))  #CHANGED
        return population  # Unchanged


    def initialize_population_v2(self):  #CHANGED
        ''' Generate a population using Latin Hypercube Sampling (LHS)
            N = popsize solutions, each solution containing
            N = individual_dims genes, each gene within <gene_limits>'''  # Keeping your comment unchanged
        
        # Create a Latin Hypercube sampler  # Keeping your comment unchanged
        sampler = qmc.LatinHypercube(d=self.individual_dims)  #CHANGED

        # Generate a sample  # Keeping your comment unchanged
        lhs_sample = sampler.random(n=self.popsize)  #CHANGED

        # Scale the sample to the range [-1, 1]  # Keeping your comment unchanged
        population = qmc.scale(lhs_sample, l_bounds=self.gene_limits[0], u_bounds=self.gene_limits[1])  #CHANGED
        
        return population  # Unchanged


    # promoting diversity
    def vectorized_fitness_sharing(self, fitnesses: np.ndarray, population: np.ndarray, k=0.05) -> np.ndarray:  #CHANGED
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
        # Calculate the max possible distance between two genes  # Keeping your comment unchanged
        gene_distance_max = self.gene_limits[1] - self.gene_limits[0]  #CHANGED
        
        # Calculate max possible distance between two individuals and sigma_share  # Keeping your comment unchanged
        max_possible_distance = gene_distance_max * np.sqrt(population.shape[1])  # Unchanged
        sigma_share = k * max_possible_distance  # Unchanged

        # Calculate pairwise Euclidean distances for the entire population  # Keeping your comment unchanged
        diff_matrix = population[:, np.newaxis] - population[np.newaxis, :]  # Unchanged
        distances = np.linalg.norm(diff_matrix, axis=2)  # Unchanged

        # Apply the niche count function where distance < sigma_share  # Keeping your comment unchanged
        niche_matrix = np.where(distances < sigma_share, 1 - (distances / sigma_share), 0)  # Unchanged

        # Each individual includes its own contribution (sh(d) = 1 for distance 0 with itself)  # Keeping your comment unchanged
        np.fill_diagonal(niche_matrix, 1)  # Unchanged
        
        # Calculate the niche count for each individual  # Keeping your comment unchanged
        niche_counts = niche_matrix.sum(axis=1)  # Unchanged

        # Calculate shared fitnesses  # Keeping your comment unchanged
        shared_fitnesses = fitnesses / niche_counts  # Unchanged

        if all(shared_fitnesses < 100) == False:
            breakpoint()

        return shared_fitnesses  # Unchanged


    def vectorized_tournament_selection(self, population, fitnesses, n_tournaments, k=15):  #CHANGED
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
        # Randomly select k individuals for each tournament (by default with replacement)  # Keeping your comment unchanged
        players = np.random.choice(np.arange(len(population)), size=(n_tournaments, k))  # Unchanged

        # Find the best individual (highest fitness) in each tournament  # Keeping your comment unchanged
        best_indices = np.argmax(fitnesses[players], axis=1)  # Unchanged

        # Retrieve the winning individuals (parents) from the population  # Keeping your comment unchanged
        selected_parents = population[players[np.arange(n_tournaments), best_indices]]  # Unchanged
        
        return selected_parents  # Unchanged


    def vectorized_ranking_selection(self, population, fitnesses, n_parents, beta=0.1):  #CHANGED
        """
        Selects parents using Exponential Rank-Based Selection.
                
        Parameters:
        - population: Array of individuals (each row is an individual).
        - fitnesses: Array of fitness values for the population (same length as population).
        - n_parents: Number of parents to select.
        - beta: Controls the steepness of the exponential ranking (default=0.5).
        
        Returns:
        - selected_parents: Array of selected parent individuals from the population.
        """
        # Step 1: Rank individuals based on fitness (highest fitness gets rank 0)  # Keeping your comment unchanged
        ranked_indices = np.argsort(fitnesses)[::-1]  # Indices sorted by fitness in descending order  # Unchanged
        
        # Step 2: Assign exponential selection probabilities based on ranks  # Keeping your comment unchanged
        ranks = np.arange(len(fitnesses))  # Rank values from 0 to population_size - 1  # Unchanged
        probabilities = (1 - np.exp(-beta)) * np.exp(-beta * ranks)  # Unchanged
        
        # Normalize probabilities to sum to 1  # Keeping your comment unchanged
        probabilities /= np.sum(probabilities)  # Unchanged
        
        # Step 3: Select parents based on the computed probabilities  # Keeping your comment unchanged
        selected_indices = np.random.choice(ranked_indices, size=n_parents, p=probabilities, replace=True)  # Unchanged
        
        # Step 4: Return the selected parents  # Keeping your comment unchanged
        selected_parents = population[selected_indices]  # Unchanged
        
        return selected_parents  # Unchanged


    def vectorized_parent_selection(self, population, fitnesses, n_children=2, k=15):  #CHANGED
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
        n_parents = int(len(population) / n_children) * n_children  # Ensure multiple of n_children  # Unchanged
        
        # Perform tournament selection for all parents at once  # Keeping your comment unchanged
        g_parents = self.vectorized_tournament_selection(population, fitnesses, n_parents, k)  #CHANGED
        
        # Vectorized fitness evaluation of selected parents  # Keeping your comment unchanged
        f_parents = self.evaluate_fitnesses(g_parents)  #CHANGED
        
        return g_parents, f_parents  # Unchanged


    # Survivor selection with elitism and random diversity preservation
    def survivor_selection(self, parents, parent_fitnesses, 
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


    def vectorized_blend_recombination(self, p1: np.ndarray, p2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
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


    def vectorized_crossover(self, all_parents: np.ndarray, p_crossover: float, recombination_operator: callable) -> np.ndarray:
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


    def vectorized_uncorrelated_mut_one_sigma(self, individual: np.ndarray, sigma: float, 
                                              mutation_rate: float) -> tuple[np.ndarray, float]:
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


    def gain_diversity(self, env, best_sol, population_genes):  #CHANGED
        '''
        Obtains gain for best solution in a generation and population diversity of a given generation (average genewise STD)
        '''
        # Gain  # Keeping your comment unchanged
        _, pl, el, _ = run_game(env, best_sol, test=True)  #CHANGED
        gain = pl - el  # Unchanged

        # Diversity: rows = individuals, columns = genes  # Keeping your comment unchanged
        genewise_stds = np.std(population_genes, axis=0)  # per gene, across individuals  # Unchanged
        diversity = np.mean(genewise_stds)  # Unchanged

        return gain, diversity  # Unchanged

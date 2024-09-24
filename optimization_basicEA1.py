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
        experiment_name = 'basic_' + args.exp_name
    else:
        experiment_name = 'basic_' + input("Enter Experiment (directory) Name:")
    
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
        basic_ea(popsize, mg, mr, cr, n_hidden, experiment_name,
                 env)
    else:
        basic_ea(popsize, mg, mr, cr, n_hidden, experiment_name,
                 env)

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
    return fitnesses

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
    return population.tolist()

# useful for probabilistic selection, for tournament doesnt matter
def normalize_fitness (one_fitness, all_fitnesses):
    ''''Normalizes fitness ONE fitness value between 0 and 1
            the funtion is supposed to be mapped to the whole array of fitness values'''
    if max(all_fitnesses) - min(all_fitnesses) > 0:
        normalized_fitness = (one_fitness - min(all_fitnesses)) / (max(all_fitnesses) - min(all_fitnesses))
    else:
        normalized_fitness = 0

    # prevent negative or 0 fitness values
    if normalized_fitness <= 0:
        normalize_fitness = 0.0000000001
    return normalized_fitness

# promoting diversity
def fitness_sharing(fitnesses: list[float], population: list[list], 
                    gene_limits:list[float, float], k = 0.15):
    """Apply fitness sharing as described in the slide to promote diversity and niche creation."""
    # Calculate the max possible distance between two genes
    gene_distance_max = gene_limits[1] - gene_limits[0]  
    # Calculate max possible distance between two individuals and sigma_share
    max_possible_distance = gene_distance_max * np.sqrt(len(population[0])) 
    # Scaling factor k controls fitness sharing radius
    sigma_share = k * max_possible_distance  

    shared_fitnesses = []
    for i, individual_i in enumerate(population):
        niche_count = 0
        for j, individual_j in enumerate(population):
            if i != j:  # No need to compare with self
                distance = np.linalg.norm(np.array(individual_i) - np.array(individual_j)) 
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
    players = np.random.choice(np.arange(len(population)), size=k)
    best_individual_idx = max(players, key=lambda i: fitnesses[i])
    return population[best_individual_idx]

# Parent selection
def parent_selection(population, fitnesses, env:Environment, n_children = 2):
    '''Tournament-based parent selection (for now)'''    
    n_parents = int(len(population) / n_children)
    # genotypes of parents, fitnesses of those genotypes
    g_parents = []
    for _ in range(n_parents):
        for _ in range(n_children):
            parent = tournament_selection(population, fitnesses)
            g_parents.append(parent)
    # (parallelized) fitness evaluation
    f_parents = evaluate_fitnesses(env, g_parents)
    return g_parents, f_parents

# Survivor selection with elitism and random diversity preservation
def survivor_selection(parents, parent_fitnesses, 
                       offspring, offspring_fitnesses, elite_fraction=0.8):
    """Select survivors using elitism with some randomness to maintain diversity."""
    # Combine parents and offspring
    total_population = parents + offspring
    total_fitnesses = parent_fitnesses + offspring_fitnesses
    # Sort by fitness in descending order
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
    return [ind for _, ind in survivors], [fit for fit, _ in survivors]

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
    return offspring

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

def basic_ea (popsize:int, max_gen:int, mr:float, cr:float, n_hidden_neurons:int,
              experiment_name:str, env:Environment, new_evolution:bool = True):
    ''' Basic evolutionary algorithm to optimize the weights '''
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

        # stagnation prevention
        stagnation = 0
        starting_mutation_rate, starting_crossover_rate = mr, cr

        # saves results for first pop
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')
        print( '\n GENERATION '+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '
              +str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
        file_aux.write('\n'+str(ini_g)+' '+str(round(best_fitness,6))+' '+str(round(mean_fitness,6))+' '+str(round(std_fitness,6))+' '+str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
        file_aux.close()

    # evolution loop
    for i in range(max_gen):
        # niching (fitness sharing)
        shared_fitnesses = fitness_sharing(fitnesses, population, gene_limits)
        # shared_fitnesses = fitnesses # disables fitness sharing
        # ?? crowding ??
        # ?? speciation - islands ??

        # Parent selection: (Tournament? - just first try) Probability based - YES
        parents, parent_fitnesses = parent_selection(population, shared_fitnesses, env)

        # crossover / recombination: Whole Arithmetic (basic) | Blend Recombination (best)
        offspring = crossover(parents, cr, blend_recombination)
        
        # mutation: Uncorrelated mutation with N step sizes
        offspring_mutated, sigma_primes = zip(*[uncorrelated_mut_one_sigma(ind, sigma_prime, mr) for ind in offspring])
        offspring_fitnesses = evaluate_fitnesses(env, offspring_mutated)
        sigma_prime = sigma_primes[np.argmax(offspring_fitnesses)]

        # Survivor selection with elitism & some randomness
        population, fitnesses = survivor_selection(parents, parent_fitnesses, 
                                                   list(offspring_mutated), offspring_fitnesses)

        # Check for best solution
        best_idx = np.argmax(fitnesses)
        generational_fitness = fitnesses[best_idx]
        gen_mean = np.mean(fitnesses)
        gen_std = np.std(fitnesses)
        
        best_fitness = fitnesses[best_idx]
        best_individual = population[best_idx]
        mean_fitness = np.mean(fitnesses)
        # if fitnesses[best_idx] > best_fitness:
        #     best_fitness = fitnesses[best_idx]
        #     best_individual = population[best_idx]
        #     mean_fitness = np.mean(fitnesses)
        #     std_fitness = np.std(fitnesses)
        #     stagnation = 0 # reset stagnation
        #     mr, cr = starting_mutation_rate, starting_crossover_rate
        
        # else:
        #     stagnation += 1
        #     if stagnation < 10:
        #         mr += .02
        #         cr += 0.03
        #         sigma_prime += 0.03
        #     elif stagnation >= 10 and stagnation < 20:
        #         mr += .03
        #         cr += 0.05
        #         sigma_prime += 0.06
        #     else:
        #         # too long stagnation, need new blood
        #         new_blood = initialize_population(popsize//3, individual_dims,
        #                                           gene_limits)
        #         new_fitnesses = evaluate_fitnesses(env, new_blood)

        #         # replace a third of population with new blood
        #         population[-(popsize // 3):] = new_blood
        #         fitnesses[-(popsize // 3):] = new_fitnesses
    
        #         stagnation = 0 # reset stagnation
        #         mr, cr = starting_mutation_rate, starting_crossover_rate
        #         sigma_prime = 0.05
        #         print('-----New Blood!-----')

        # OUTPUT: weights + biases vector
        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        # file_aux.write('\n\ngen best mean std sigma_prime mutation_r crossover_r')
        print( '\n GENERATION '+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '
              +str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
        file_aux.write('\n'+str(i)+' '+str(round(generational_fitness,6))+' '+str(round(gen_mean,6))+' '+str(round(gen_std,6))+' '+str(round(sigma_prime, 6))+' '+str(round(mr, 6))+' '+str(round(cr, 6)))
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','a')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt',best_individual)

        if not os.path.exists('basic_solutions'):
            os.makedirs('basic_solutions')
        np.savetxt(f'basic_solutions/{env.enemyn}best.txt', best_individual)

        # saves simulation state
        solutions = [population, fitnesses]
        env.update_solutions(solutions)
        env.save_state()

    fim = time.time() # prints total execution time for experiment
    print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
    print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


    file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log() # checks environment state

if __name__ == '__main__':
    main()
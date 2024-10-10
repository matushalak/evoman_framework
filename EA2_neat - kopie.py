###############
# @matushalak
# code adapted from NEAT-python package, XOr & OpenAI-Lander examples
# eg. https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
###############
import multiprocessing
import os
import neat
from evoman.environment import Environment
from demo_controller import player_controller
from neat_controller import neat_controller
from time import time
from pandas import DataFrame
import argparse
import pickle as pkl
import optuna

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 100, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    parser.add_argument('-trials', '--num_trials', type=int, required=False, default=100, help='Number of bayesian optimization trials') 


    return parser.parse_args()

args = parse_args()
global env, cfg, name, enemies, multi, maxgen
cfg = 'neat_config.txt'
maxgen = args.maxgen
enemies = args.enemies
multi  = 'yes' if args.multi == 'yes' else 'no'

if isinstance(args.exp_name, str):
    name = 'neat_' + args.exp_name
else:
    name = 'neat_' + input("Enter Experiment (directory) Name:")

# add enemy names
name = name + '_' + f'{str(enemies).strip('[]').replace(',', '').replace(' ', '')}'
if not os.path.exists(name):
    os.makedirs(name)

env = Environment(experiment_name=name,
                enemies=enemies,
                multiplemode=multi, 
                playermode="ai",
                player_controller=neat_controller(config_path=cfg), # you  can insert your own controller here
                enemymode="static",
                level=2,
                speed="fastest",
                visuals=False)

os.environ["SDL_VIDEODRIVER"] = "dummy"

def eval_genome(genome,config):
    '''
    Parallelized version
    '''
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness ,p,e,t = env.play(pcont=net)
    return fitness

def save_stats(StatsReporter):
    results = DataFrame({'gen':list(range(maxgen)),
                         'best':StatsReporter.get_fitness_stat(max),
                         'mean':StatsReporter.get_fitness_mean(),
                         'sd':StatsReporter.get_fitness_stdev(),
                         'med':StatsReporter.get_fitness_median(),
                         'worst':StatsReporter.get_fitness_stat(min)})
    results.to_csv(name + '/results.txt')

def run(config_path):
    start = time()
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # population
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run for N generations
    # parallel
    parallel_evals = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(parallel_evals.evaluate, maxgen)

    # winner = pop.run(eval_genomes, 50) # classic
    winn_gene = stats.best_genome()
    winner_net = neat.nn.FeedForwardNetwork.create(winn_gene, config)

    # save results
    save_stats(stats)
    # save controller
    with open(name + '/best.pkl', 'wb') as f:
        pkl.dump(winner_net, f)

    end = time()-start
    print(end)

    # Display winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    return 0 #RETURN FINAL VALUE FOR FITNESS!


# -------------------------------------------------------- BAYESIAN

def objective(trial, popsize, mg, n_hidden, experiment_name,
                 env, save_gens, num_reps):
    # Hyperparameter search space
    scaling_factor = trial.suggest_float('scaling_factor', 0.01, 0.2)  # Fitness sharing scaling factor
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.75)  # Mutation rate
    sigma_prime = trial.suggest_float('sigma_prime', 0.01, 1.0)  # Mutation sigma
    crossover_rate = trial.suggest_float('crossover_rate', 0.05, 0.75)  # Crossover rate
    alpha = trial.suggest_float('alpha', 0.1, 2.0)  # Recombination factor
    tournament_size = trial.suggest_int('tournament_size', 2, 10)  # Tournament selection size
    elite_fraction = trial.suggest_float('elite_fraction', 0.1, 0.9)  # Elite fraction in survivor selection

    hyperparameters = (scaling_factor, sigma_prime, alpha, tournament_size, elite_fraction, mutation_rate, crossover_rate,
                       popsize, mg)

    # Pass the hyperparameters as arguments
    performance = mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps)
    
    return performance 

def mean_result_EA1(hyperparameters, n_hidden, experiment_name,
                                    env, save_gens, num_reps):

    avg_fitness = 0
    for i in range(num_reps):
        print(f'Starting repetition number: {i}')
        fitness = basic_ea(hyperparameters, n_hidden, experiment_name,
                    env, save_gens)
        avg_fitness += fitness

    avg_fitness = avg_fitness/num_reps

    return avg_fitness

def bayesian_optimization(num_trials, config_path=cfg):

    num_reps = 3
    
    study = optuna.create_study(direction='maximize',
                                        storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
                                        study_name="08_10_nightrun")  # If you want to maximize the fitness score
    study.optimize(lambda trial: objective(trial, config_path, num_reps), n_trials=num_trials)

    return 0


if __name__ == '__main__':
    num_trials = args.num_trials
    bayesian_optimization(num_trials) 
    run(config_path=cfg)    
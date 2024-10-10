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

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 10, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    
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

if __name__ == '__main__':
    run(config_path=cfg)    
###############
# @matushalak
# code adapted from NEAT-python package, XOr & OpenAI-Lander examples
# eg. https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
###############
import multiprocessing
import os
import neat
from evoman.environment import Environment
#from demo_controller import player_controller
from neat_controller import neat_controller
from time import time
from pandas import DataFrame
import argparse
import pickle as pkl

from EA2_NEAT_optimizer import NEAT

def parse_args():
    '''' Function enabling command-line arguments'''
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Optimize weights of Controller NN using EA")

    # Define arguments
    parser.add_argument('-name', '--exp_name', type=str, required=False, help="Experiment name")
    parser.add_argument('-mg', '--maxgen', type=int, required=False, default = 100, help="Max generations (eg. 500)")
    parser.add_argument('-nmes', '--enemies', nargs='+', type=int, required=False, default=[5, 6], help='Provide list(s) of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    #EA1_line_plot_runs   
    parser.add_argument('-dir', '--directory', type=str, default='1510_EA2_neat_line_plot_runs', required=False, help="Directory to save runs")
    parser.add_argument('-nruns', '--num_runs', type=int, required=False, default=50, help="Number of repetitive neat runs")  # Unchanged
    
    # XXX NEAT specific
    parser.add_argument('-cfg', '--config', type=str, required=True, help="Directory with optimized config")
    

    return parser.parse_args()

global cfg, name, enemy_set, base_dir, num_runs, args
args = parse_args()
cfg = args.config
enemy_set = args.enemies
base_dir = args.directory
num_runs = args.num_runs


def multiple_runs():
    # Create the base directory if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    enemy_dir = os.path.join(base_dir, f'EN{enemy_set}')
    if not os.path.exists(enemy_dir):
        os.makedirs(enemy_dir)

    for run_i in range(1, num_runs+1):  # Run the EA e.g. 10 times for each enemy
        run_dir = os.path.join(enemy_dir, f'run_{run_i}_EN{enemy_set}')
        # Run the ClassicEA for this enemy and run number
        print(f'\nRunning EA2 (neat) for enemy (set) {enemy_set}, run {run_i}\n')
        Neat = NEAT(args, run_dir, cfg)
        fitness = Neat.run()



if __name__ == '__main__':
    multiple_runs()    
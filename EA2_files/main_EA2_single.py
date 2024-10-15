###############
# @matushalak
# code adapted from NEAT-python package, XOr & OpenAI-Lander examples
# eg. https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
###############
import multiprocessing
import os
import neat
from evoman.environment import Environment
# from demo_controller import player_controller
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
    parser.add_argument('-nmes', '--enemies', nargs = '+', type = int, required=True, default = False, help='Provide list of enemies to train against')
    parser.add_argument('-mult', '--multi', type=str, required=False, default = 'yes', help="Single or Multienemy")
    
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = 'neat_config.txt'
    enemies = args.enemies

    if isinstance(args.exp_name, str):
        name = 'neat_' + args.exp_name
    else:
        name = 'neat_' + input("Enter Experiment (directory) Name:")

    # add enemy names
    name = name + '_' + f'{str(enemies).strip('[]').replace(',', '').replace(' ', '')}'

    if not os.path.exists(name):
        os.makedirs(name)

    Neat = NEAT(args, name, cfg)
    fitness = Neat.run()
      

if __name__ == '__main__':
        main()
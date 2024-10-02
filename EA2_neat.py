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

cfg = 'neat_config.txt'
name = 'neattest'
enemies = [4]
test_enemies = [4]
multi = 'no'

if not os.path.exists(name):
        os.makedirs(name)

global env
env = Environment(experiment_name=name,
                    enemies=enemies,
                    multiplemode=multi, 
                    playermode="ai",
                    player_controller=neat_controller(config_path=cfg), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)

# def eval_genomes(population_genomes,config):
#     for _, genome in population_genomes:
#         net = neat.nn.FeedForwardNetwork.create(genome, config)
#         fitness ,p,e,t = env.play(pcont=net)
#         genome.fitness = fitness
def eval_genome(genome,config):
    '''
    Parallelized version
    '''
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness ,p,e,t = env.play(pcont=net)
    return fitness



def run(config_path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
							neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # population
    pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(25,900))

    # Run for N generations
    # parallel
    parallel_evals = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(parallel_evals.evaluate, 50)

    # winner = pop.run(eval_genomes, 50) # classic
    winn_gene = stats.best_genome()
    winner_net = neat.nn.FeedForwardNetwork.create(winn_gene, config)

    # Display winning genome
    print('\nBest genome:\n{!s}'.format(winner))
    # play winning genome
    env.update_parameter('speed','normal')
    env.update_parameter('visuals', True)
    env.update_parameter('multiplemode', 'no')
    # test against enemies
    for en in test_enemies:
        #Update the enemy
        env.update_parameter('enemies',[en])
        f, pl, el, t = env.play(pcont = winner_net)
        print(f, pl, el, t)
        print(f'Gain: {pl-el}')

if __name__ == '__main__':
    run(config_path=cfg)
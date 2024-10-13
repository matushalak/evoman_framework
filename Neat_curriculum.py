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

class NEAT:
    def __init__(self):
        args = parse_args()
        self.cfg = 'neat_config.txt'
        self.maxgen = args.maxgen
        self.enemies = args.enemies
        self.multi  = 'yes' if args.multi == 'yes' else 'no'

        if isinstance(args.exp_name, str):
            self.name = 'neat_' + args.exp_name
        else:
            self.name = 'neat_' + input("Enter Experiment (directory) Name:")

        # add enemy names
        self.name = self.name + '_' + f'{str(self.enemies).strip('[]').replace(',', '').replace(' ', '')}'
        if not os.path.exists(self.name):
            os.makedirs(self.name)

        os.environ["SDL_VIDEODRIVER"] = "dummy"

    def save_stats(self, stats, filename):
        '''
        Saves the evolutionary statistics to a CSV file.
        '''
        results = DataFrame({
            'gen': list(range(len(stats.most_fit_genomes))),
            'best': [c.fitness for c in stats.most_fit_genomes],
            'mean': stats.get_fitness_mean(),
            'sd': stats.get_fitness_stdev()
        })
        results.to_csv(filename, index=False)

    def eval_genome(self, genome,config):
        '''
        Parallelized version
        '''
        env = Environment(experiment_name=self.name,
                    enemies=self.enemies,
                    multiplemode=self.multi, 
                    playermode="ai",
                    player_controller=neat_controller(config_path=self.cfg), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False)
        
        env.update_parameter('enemies', self.enemies)

        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness ,p,e,t = env.play(pcont=net)
        return float(p - e) # min player life - max enemy life : modified Gain measure
        # to focus on beating enemies
        # return float(100 - e)
        # return 0.9*(100 - e) + 0.1*(100 - p)

    def run(self, config_path):
        start = time()
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        
        # Initial population
        pop = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)

        # Different enemy sets = different 'curricula', starting from easy to hard
        # TODO REVISE CURRICULA!!!
        curriculum = {1:{
                        'enems':[1,7],
                        'gens':self.maxgen},
                    2:{
                        'enems':[1,7,3,5],
                        'gens':self.maxgen},
                    3:{
                        'enems':[1,7,3,5,8],
                        'gens':self.maxgen},
                    4:{
                        'enems':[4,1,7,3,5,8],
                        'gens':self.maxgen},
                    5:{
                        'enems':[4,1,7,3,5,8,6],
                        'gens':self.maxgen}
                        }

        # NOTE CURRICULUM LEARNING
        for stage, curriculum in curriculum.items():
            print(f"\nStarting Stage {stage} with enemies {curriculum['enems']}")

            #update env with new enemies
            self.enemies = curriculum['enems']

            # Parallel evaluator
            parallel_evals = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.eval_genome)

            # TODO add expert solutions to initial population in stages
            # ---------------------------

            # Run for N generations
            pop.run(parallel_evals.evaluate, curriculum['gens'])

            winn_gene = stats.best_genome()
            winner_net = neat.nn.FeedForwardNetwork.create(winn_gene, config)
            with open(self.name + f'/stage{stage}_best.pkl', 'wb') as f:
                pkl.dump(winner_net, f)

            # save results
            self.save_stats(stats, self.name + f'/stage{stage}_stats.csv')

        # save controller
        with open(self.name + '/best.pkl', 'wb') as f:
            pkl.dump(winner_net, f)

        end = time()-start
        print(end)

        # Display winning genome
        print('\nBest genome:\n{!s}'.format(winn_gene))

if __name__ == '__main__':
    Neat = NEAT()
    Neat.run(Neat.cfg)
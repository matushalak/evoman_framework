# Evolutionary algorithm project

This repository contains code for running and analyzing two different evolutionary algorithms (EAs) for a multi-objective optimization task (designing a generalist controller in the Evoman framework).

## Correspondence with report
- **EA1** corresponds to **Baseline EA** in the report.
- **EA2** corresponds to **Neuroevolution of Augmenting Topologies algorithm (NEAT)** in the report.

## Algorithm Scripts
The scripts that do the heavy lifting and contain the class implemeting each algorithm, called by other scripts:
- In `EA1_files`: `EA1_optimizer.py` contains the code for the Baseline EA class

- In `EA2_files`: `EA2_NEAT_optimizer.py` contains the code for the NEAT EA class

## Running the algorithms
To run the evolutionary algorithm once, execute the following scripts, with the appropriate command-line arguments:
- In `EA1_files`: `main_EA1_single.py` for EA1 (Baseline EA)

- In `EA2_files`: `main_EA2_single.py` for EA2 (NEAT)

## Running multiple RUNS
To run the optimization n times for any enemy set and create the corresponding report folders, use:
- In `EA1_files`: `main_EA1_multiple.py` for EA1 (Baseline EA)

- In `EA2_files`: `main_EA2_multiple.py` for EA2 (NEAT)

## Bayesian Parameter optimization
The following scripts perform parameter optimizations for the algorithms:
- In `EA1_files`:`bayesian_EA1.py` for EA1 (Baseline EA)

- In `EA2_files`: `bayesian_EA2.py` for EA2 (NEAT)
## Data retrieval and preparation
To retrieve the data from the report results and prepare the gain data for boxplots, use:
- In `EA1_files`: `get_gain_EA1_v2.py` for EA1 (Baseline EA)

- In `EA2_files`: `get_gain_EA2_NEAT_v2.py` for EA2 (NEAT)
## Analysis
The following scripts are available for analysis:
- `analyse_fitness_v2.py`: analyzes fitness and diversity, creating a 2x2 grid of line plots for mean fitnesses and distribution of best fitness in each run  for both EAs & both enemy sets. It also performs statistical tests on the data.
- `analyse_gain.py`: takes data from the folder `gain_res_EA1` or `gain_res_EA2`, creates a boxplot, and performs statistical tests on the gain data.

## Controller scripts
### Running solutions on Evoman
- `controller_task2.py` 
### Constructing Controllers for Evoman framework
- `demo_controller.py` & `neat_controller.py`: these scripts can be used to construct a controller for the evoman framework based on EA1 & NEAT output.

## Folder structure
- `EA1_files`: contains results & all files for EA1 (Baseline EA).
- `EA2_files`: contains results & all files for EA2 (NEAT).
- `figures`: contains all figures used in the Task 2 Report
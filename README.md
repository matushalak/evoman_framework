# Evolutionary algorithm project

This repository contains code for running and analyzing two different evolutionary algorithms (EAs).

## Correspondence with report
- **EA1** corresponds to **Baseline EA** in the report.
- **EA2** corresponds to **IM-EA** in the report.

## Running the algorithms
To run the evolutionary algorithm once, execute the following scripts:
- `optimization_basicEA1.py` for EA1 (Baseline EA)
- `optimization_basicEA2.py` for EA2 (IM-EA)

## Running multiple trials
To run the optimization 10 times for 3 different enemies and create the corresponding report folders, use:
- `optimization_basicEA1_10runs.py` for EA1
- `optimization_basicEA2_10runs.py` for EA2

## Parameter optimization
The following scripts perform parameter optimizations for the algorithms:
- `grid_search_EA1.py` for EA1
- `grid_search_EA2.py` for EA2

## Data retrieval and preparation
To retrieve the data from the report results and prepare the gain data for boxplots, use:
- `get_gain_data.py`

## Analysis
The following scripts are available for analysis:
- `analyse_fitness_and_diversity.py`: analyzes fitness and diversity, creating a 3x2 grid of line plots for mean fitnesses and mean diversities for both EAs. It also performs statistical tests on the data.
- `analyse_gain.py`: takes data from the folder `gain_results` (created by `get_gain_data.py`), creates a boxplot, and performs statistical tests on the gain data.

## Controller scripts
- `controller_task1.py` and `demo_controller.py`: these scripts can be used as a controller for specific data.

## Folder structure
- `EA1_report_results`: contains results for EA1 (Baseline EA).
- `EA2_report_results`: contains results for EA2 (IM-EA).
- `gain_results`: folder created by `get_gain_data.py`.

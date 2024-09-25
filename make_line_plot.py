import os
import pandas as pd
import matplotlib.pyplot as plt

#set manual input:
enemy = 2
#set the directory where the 10 runs are stored
enemy_folder = os.path.join('basic_results_for_plotting', f"enemy_{enemy}", 'pop_100_mg_50')

#now, the script starts from: if __name__ == "__main__"


def process_results_for_enemy(enemy_folder, enemy):
    
    if not os.path.isdir(enemy_folder):
        print(f"Enemy folder not found: {enemy_folder}")
        return

    all_dfs = [] #define empty all_dfs

    #iterate through the run folders (e.g., run_1_x, run_2_x, etc.)
    for run_folder in os.listdir(enemy_folder):
        run_path = os.path.join(enemy_folder, run_folder)
        
        if os.path.isdir(run_path):
            result_file = os.path.join(run_path, 'results.txt')  #each run has a 'results' file
            
            if os.path.exists(result_file):
                #read results file
                df = pd.read_csv(result_file, delim_whitespace=True)
                df = df[pd.to_numeric(df['best'], errors='coerce').notnull()]  #clean invalid rows
                
                #convert columns to numeric
                df['best'] = pd.to_numeric(df['best'])
                df['mean'] = pd.to_numeric(df['mean'])
                df['std'] = pd.to_numeric(df['std'])
                df['gen'] = pd.to_numeric(df['gen'])

                #append DataFrame to the list
                all_dfs.append(df)
                
    return all_dfs  

            
def make_line_plot_across_gens(all_dfs):

    if all_dfs:
        #concat all dataframes
        combined_df = pd.concat(all_dfs)
    
        #group by gen and calculate mean and std for mean and best fitness
        grouped_stats = combined_df.groupby('gen').agg(
            mean_avg_fitness=('mean', 'mean'),
            std_avg_fitness=('mean', 'std'),
            mean_max_fitness=('best', 'mean'),
            std_max_fitness=('best', 'std')
        )
    
        plt.figure(figsize=(14, 12))
        generations = grouped_stats.index
    
        #plot mean fitnss with standard deviation shading
        plt.plot(generations, grouped_stats['mean_avg_fitness'], label="Average Mean Fitness", color="#1f77b4")
        plt.fill_between(generations, 
                         grouped_stats['mean_avg_fitness'] - grouped_stats['std_avg_fitness'],
                         grouped_stats['mean_avg_fitness'] + grouped_stats['std_avg_fitness'],
                         color="#1f77b4", alpha=0.1)
    
        #plot best fitness with standard deviation shading
        plt.plot(generations, grouped_stats['mean_max_fitness'], label="Average Best Fitness", color="#d62728")
        plt.fill_between(generations, 
                         grouped_stats['mean_max_fitness'] - grouped_stats['std_max_fitness'],
                         grouped_stats['mean_max_fitness'] + grouped_stats['std_max_fitness'],
                         color="#d62728", alpha=0.1)
    
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'Fitness Evolution')
        plt.legend()
        plt.grid()
    
        #save plot in the enemy folder
        #plot_file = os.path.join(enemy_folder, f'fitness_plot.png')
        #plt.savefig(plot_file)
        #print(f"Plot saved at: {plot_file}")
    
        #show plot
        plt.show()
        
    return 0
            
if __name__ == "__main__":
    all_dfs = process_results_for_enemy(enemy_folder, enemy)
    make_line_plot_across_gens(all_dfs)

import os
import json

# Set the base folder for dummy data
base_folder = 'dummy_data_box_plot'
os.makedirs(base_folder, exist_ok=True)

# Example gains data for algo1 and algo2
dummy_data_algo1_enemy_2 = {
    "run_1": [89.80, 81.40, 76.00, 81.40, 89.80],
    "run_2": [40.00, 81.40, 40.00, 40.00, 87.40],
    "run_3": [88.00, 29.80, 68.80, 29.80, 88.00],
    "run_4": [86.80, 86.80, 86.80, 27.40, 69.40],
    "run_5": [92.20, 92.20, 92.20, 92.20, 92.20],
    "run_6": [89.20, 89.20, 89.20, 89.20, 89.20],
    "run_7": [70.00, 70.00, 81.40, 70.00, 70.00],
    "run_8": [-50.00, -50.00, 80.80, 89.20, 89.20],
    "run_9": [-50.00, 71.20, 71.20, 71.20, 90.40],
    "run_10": [83.80, 88.00, -70.00, 83.80, 83.80]
}

dummy_data_algo2_enemy_2 = {
    "run_1": [98.80, 74.80, 91.60, 89.80, -17.00],
    "run_2": [68.80, 86.80, 77.20, 89.20, 87.40],
    "run_3": [89.20, 94.60, 95.20, 89.20, 88.00],
    "run_4": [98.80, 78.40, 94.60, 55.00, 88.00],
    "run_5": [76.60, 88.00, 95.80, 67.60, 67.00],
    "run_6": [89.20, 73.00, 89.80, 93.40, 58.00],
    "run_7": [91.60, 92.20, 87.40, 90.40, 89.80],
    "run_8": [88.00, 90.40, 97.60, 87.40, 84.40],
    "run_9": [94.60, 82.00, 65.80, 93.40, 93.40],
    "run_10": [-15.00, 61.00, 86.80, 100.00, 85.60]
}

# Using the same data structure for enemies 6 and 8 for simplicity
dummy_data_algo1_enemy_6 = dummy_data_algo1_enemy_2
dummy_data_algo1_enemy_8 = dummy_data_algo1_enemy_2
dummy_data_algo2_enemy_6 = dummy_data_algo2_enemy_2
dummy_data_algo2_enemy_8 = dummy_data_algo2_enemy_2

# Function to save dummy data in JSON format
def save_json_data(algo_folder, enemy, data):
    os.makedirs(algo_folder, exist_ok=True)
    file_path = os.path.join(algo_folder, f'enemy_{enemy}_gains.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved: {file_path}")

def generate_dummy_data():
    enemies = [2, 6, 8]
    algorithms = ['algo1', 'algo2']
    
    for enemy in enemies:
        for algo in algorithms:
            algo_folder = os.path.join(base_folder, algo)
            
            # Save data for each enemy and algorithm
            if algo == 'algo1':
                if enemy == 2:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_2)
                elif enemy == 6:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_6)
                elif enemy == 8:
                    save_json_data(algo_folder, enemy, dummy_data_algo1_enemy_8)
            else:  # algo2
                if enemy == 2:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_2)
                elif enemy == 6:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_6)
                elif enemy == 8:
                    save_json_data(algo_folder, enemy, dummy_data_algo2_enemy_8)

if __name__ == "__main__":
    generate_dummy_data()

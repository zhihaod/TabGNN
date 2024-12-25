import json
import argparse
from utilities import load_dataset, load_dataset_ratio
from trainer import Trainer
from GNNModel import GNNModel
import pandas as pd
import numpy as np

# Function to load config from a JSON file
def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

# Define the main function
def main(names):
    # Load configuration from the JSON file
    configs = load_config('config.json')  # Adjust the path if necessary
    res = {}
    for name in names:
        print(f"Running with config: {name}")
        if name not in configs:
            print(f"Configuration {name} not found in config.json. Skipping...")
            continue    
        config = configs[name]
       
    # Load the dataset    
        df = pd.read_csv(config['path'] + '/' + config['name'] + '/' + 'comprehensive_network0.01.csv')
        ratios = np.linspace(0,1, 51)
        res_l = []
        for ratio in ratios: 
            print(ratio)
            dataset = load_dataset_ratio(config['path'], data_name=config['name'], label_name='label', ratio = ratio)
    # Initialize trainer with dataset and configuration
            trainer = Trainer(dataset, GNNModel, config)
    # Run cross-validation
            trainer.cross_validate()
    # Print average results
            res_ = trainer.print_average_results()
            res_l.append(res_)
        res[name] = res_l
    filename = './results/results_ratio_hat.json'
    with open(filename, 'w') as json_file:
        json.dump(res, json_file) 
    print(f"Results has been written to {filename}")
    print(res)


# Execute the main function when the script is run
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run GNN model training with a specific config")
    parser.add_argument('names', nargs='+', type=str, help='The name(s) of the configuration(s) to use from the config file')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    # Pass the argument to the main function
    main(args.names)

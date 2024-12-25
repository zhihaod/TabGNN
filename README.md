# TabGNN

TabGNN is a machine learning framework designed for tabular data using graph-based methodologies. By constructing a Feature Relational Graph (FRG) from tabular datasets, TabGNN captures complex relationships between features, enabling enhanced performance and interpretability for a variety of tasks.

## Project Structure

```
TabGNN
├── Run_BuildComprehensiveNetwork.sh    # Shell script to build a comprehensive network
├── run.sh                              # Script to run the project
├── config.json                         # Configuration file for project settings
├── output.log                          # Log file for execution details
├── nohup.out                           # Log file for background processes
├── results/                            # Directory containing project results
├── data/                               # Directory for datasets
├── src/                                # Source code directory
├── .gitignore                          # Git ignore file
├── .git/                               # Git repository metadata
├── .ipynb_checkpoints/                 # Jupyter notebook checkpoints
└── .DS_Store                           # macOS system file (ignore)
```

## Features

- Constructs a Feature Relational Graph (FRG) from tabular datasets.
- Employs Information Gain (IG) and Mutual Information (MI) to define feature relationships.
- Provides configurable graph construction strategies.
- Outputs interpretable and compact graph representations.
- Supports various machine learning tasks with enhanced feature modeling.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the project by editing `config.json` as needed.

## Usage

### Running the Project


1. Build the comprehensive network using:
   ```bash
   bash Run_BuildComprehensiveNetwork.sh
   ```
2. Use the `run.sh` script to run the TabGNN:
   ```bash
   bash run.sh
   ```
### Input Data
- Place your dataset files in the `data/` directory. Ensure the format aligns with the configurations in `config.json`.

### Results
- Processed results and outputs will be saved in the `results/` directory.

## Configuration
- Modify `config.json` to adjust the parameters for graph construction, dataset paths, and other project settings.

## Dependencies
- Python 3.6+
- Required libraries are listed in `requirements.txt`.





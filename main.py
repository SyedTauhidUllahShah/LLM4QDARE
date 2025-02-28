import json
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from models import LLMModel
from evaluation import Evaluator
from visualization import Visualizer

def main():
    """
    Main function to run the LLMQDA experiment.
    """
    # Load configuration
    config_path = os.path.join('config', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Update paths in config to reflect new directory structure
    for dataset_name, dataset_info in config['datasets'].items():
        dataset_info['path'] = os.path.join('data', dataset_info['path'])
    
    # Initialize components
    use_simulation = False
    print(f"Initializing LLMQDA with configuration from {config_path}")
    print(f"Using simulation mode: {use_simulation}")
    
    # Load data
    data_loader = DataLoader(config)
    datasets = data_loader.load_datasets()
    if not datasets:
        print("No datasets loaded. Exiting.")
        return
    
    # Initialize model
    model = LLMModel(config, use_simulation=use_simulation)
    
    # Initialize evaluator
    evaluator = Evaluator(data_loader, model)
    
    # Define custom experiment parameters
    custom_params = {
        'shot_types': ['zero-shot', 'one-shot', 'few-shot'],
        'prompt_lengths': ['long'],
        'context_levels': ['full_context'],
        'sample_size': None,  # Process all samples in the dataset
        'num_runs': 1
    }
    
    # Run experiment
    results = evaluator.run_experiment(params=custom_params, config=config)
    
    # Initialize visualizer and generate outputs
    results_dir = 'results'
    visualizer = Visualizer(results)
    try:
        visualizer.plot_results(output_dir=results_dir)
    except Exception as e:
        print(f"Error in plotting: {e}")
        print("Continuing without plots...")
    
    visualizer.save_results(output_dir=results_dir)
    visualizer.generate_report(output_dir=results_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
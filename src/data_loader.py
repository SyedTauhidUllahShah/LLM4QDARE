import json
import os
import pandas as pd

class DataLoader:
    """
    Handles the loading and preprocessing of requirements datasets.
    """
    
    def __init__(self, config):
        """
        Initialize the DataLoader with configuration settings.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.datasets = {}
    
    def ensure_directory(self, directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def load_datasets(self):
        """Load the requirements datasets from CSV files defined in the configuration."""
        for dataset_name, dataset_info in self.config['datasets'].items():
            path = dataset_info['path']
            try:
                df = pd.read_csv(path)
                self.datasets[dataset_name] = df
                print(f"Loaded {dataset_name} dataset with {len(df)} requirements")
            except Exception as e:
                print(f"Error loading {dataset_name} dataset from {path}: {e}")
        return self.datasets
    
    def generate_prompt(self, requirement, dataset_name, shot_type, prompt_length, context_level):
        """
        Generate a prompt for a given requirement based on configuration, restricting output to dataset labels.
        
        Args:
            requirement: Dictionary with requirement statement and true label
            dataset_name: Name of the dataset (lms or smart_home)
            shot_type: Shot type (zero-shot, one-shot, few-shot)
            prompt_length: Prompt length (short, medium, long)
            context_level: Context level (no_context, some_context, full_context)
            
        Returns:
            Formatted prompt string
        """
        template_key = f"{shot_type}_{prompt_length}"
        template = self.config['prompt_templates'].get(template_key, "Analyze the following requirement: {requirement}\n\nLabel:")
        
        prompt = template.replace("{requirement}", requirement['requirement'])
        system_type = "Library Management" if dataset_name == "lms" else "Smart Home"
        prompt = prompt.replace("{system_type}", system_type)
        
        context = ""
        if context_level != "no_context":
            context = self.config['datasets'][dataset_name]['context']
            if context_level == "full_context":
                context = self.config['datasets'][dataset_name]['detailed_context']
        prompt = prompt.replace("{context}", context)
        
        # Get valid labels from the dataset, handling NaN and ensuring strings
        if dataset_name in self.datasets:
            valid_labels = self.datasets[dataset_name]['Label'].dropna().astype(str).unique()
            valid_labels = sorted(valid_labels)  # Sort after converting to strings
        else:
            # Fallback to config examples if dataset not loaded
            valid_labels = [ex['label'] for ex in self.config['examples'][dataset_name]['one-shot']]
            valid_labels = sorted(valid_labels)
        valid_labels_str = ", ".join(valid_labels)
        
        # Add valid labels instruction
        if shot_type == "zero-shot":
            prompt = prompt.replace("Label:", f"Label (choose only from: {valid_labels_str}):")
        
        # Handle examples for one-shot and few-shot prompts
        if shot_type != "zero-shot":
            examples = self.config['examples'][dataset_name][shot_type]
            
            if shot_type == "one-shot":
                prompt = prompt.replace("{example_requirement}", examples[0]['requirement'])
                prompt = prompt.replace("{example_label}", examples[0]['label'])
                prompt = prompt.replace("Label:", f"Label (choose only from: {valid_labels_str}):")
            
            elif shot_type == "few-shot":
                # Use first three examples
                for i in range(3):
                    if i < len(examples):
                        prompt = prompt.replace(f"{{example{i+1}_requirement}}", examples[i]['requirement'])
                        prompt = prompt.replace(f"{{example{i+1}_label}}", examples[i]['label'])
                    else:
                        prompt = prompt.replace(f"Requirement: {{example{i+1}_requirement}}\nLabel: {{example{i+1}_label}}\n", "")
                prompt = prompt.replace("Label:", f"Label (choose only from: {valid_labels_str}):")
        
        return prompt
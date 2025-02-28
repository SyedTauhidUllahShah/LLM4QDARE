import time
import json
import os

class Evaluator:
    """
    Handles the evaluation of requirement labeling performance.
    """
    
    def __init__(self, data_loader, model):
        """
        Initialize the evaluator.
        
        Args:
            data_loader: DataLoader instance
            model: LLMModel instance
        """
        self.data_loader = data_loader
        self.model = model
        self.results = {}
    
    def evaluate_requirements(self, dataset_name, shot_type, prompt_length, context_level, sample_size=None, checkpoint_file=None, checkpoint_interval=10):
        """
        Evaluate requirements with the specified configuration.
        
        Args:
            dataset_name: The dataset name
            shot_type: The shot type
            prompt_length: The prompt length
            context_level: The context level
            sample_size: Optional sample size to limit processing
            checkpoint_file: File to save progress checkpoints
            checkpoint_interval: Interval for saving checkpoints
            
        Returns:
            Dictionary of evaluation results
        """
        if dataset_name not in self.data_loader.datasets:
            print(f"Dataset {dataset_name} not loaded")
            return {}
        
        dataset = self.data_loader.datasets[dataset_name]
        if sample_size and sample_size < len(dataset):
            dataset = dataset.sample(sample_size, random_state=42)
        
        print(f"Evaluating {dataset_name} with {shot_type}, {prompt_length} prompts, {context_level}")
        
        # Check for existing checkpoint
        true_labels = []
        pred_labels = []
        raw_responses = []
        start_idx = 0
        
        # Setup checkpoint file
        if checkpoint_file is None and checkpoint_interval:
            results_dir = 'results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            checkpoint_file = os.path.join(results_dir, f"checkpoint_{dataset_name}_{shot_type}_{prompt_length}_{context_level}.json")
        
        # Try to load checkpoint
        if checkpoint_file and os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    true_labels = checkpoint_data.get('true_labels', [])
                    pred_labels = checkpoint_data.get('pred_labels', [])
                    raw_responses = checkpoint_data.get('raw_responses', [])
                    start_idx = len(true_labels)
                    print(f"Resuming from checkpoint, starting at index {start_idx}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        
        total = len(dataset)
        dataset_rows = list(dataset.iterrows())
        
        for idx, (_, row_data) in enumerate(dataset_rows[start_idx:], start=start_idx):
            if idx % 10 == 0:
                print(f"Processing {idx}/{total} requirements...", end='\r')
            
            if 'Requirement Statement' not in row_data or 'Label' not in row_data:
                req_col = next((c for c in row_data.index if 'requirement' in c.lower()), None)
                label_col = next((c for c in row_data.index if 'label' in c.lower()), None)
                if req_col and label_col:
                    requirement = {'requirement': row_data[req_col], 'label': row_data[label_col]}
                else:
                    print(f"\nError: Could not find requirement or label columns in dataset.")
                    print(f"Available columns: {row_data.index.tolist()}")
                    return {}
            else:
                requirement = {'requirement': row_data['Requirement Statement'], 'label': row_data['Label']}
            
            prompt = self.data_loader.generate_prompt(requirement, dataset_name, shot_type, prompt_length, context_level)
            
            if self.model.use_simulation:
                response = self.model.simulate_response(requirement['label'], dataset_name, shot_type, self.data_loader.datasets)
                cleaned_response = response
            else:
                response = self.model.generate_response(prompt)
                cleaned_response = self.model.clean_response(response)
                time.sleep(0.5)  # Avoid rate limits
            
            true_labels.append(requirement['label'])
            pred_labels.append(cleaned_response)
            raw_responses.append(response)
            
            # Save checkpoint at regular intervals
            if checkpoint_file and checkpoint_interval and (idx + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'true_labels': true_labels,
                    'pred_labels': pred_labels,
                    'raw_responses': raw_responses
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
                print(f"\nCheckpoint saved at index {idx + 1}/{total}")
        
        print(f"Processed {total}/{total} requirements.      ")
        
        if not self.model.use_simulation and raw_responses:
            print("\nSample of responses:")
            for i in range(min(3, len(raw_responses))):
                print(f"True: '{true_labels[i]}'")
                print(f"Raw: '{raw_responses[i]}'")
                print(f"Cleaned: '{pred_labels[i]}'")
                print()
        
        metrics = self.calculate_metrics(true_labels, pred_labels)
        
        # Save final checkpoint
        if checkpoint_file:
            checkpoint_data = {
                'true_labels': true_labels,
                'pred_labels': pred_labels,
                'raw_responses': raw_responses,
                'metrics': metrics
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"Final results saved to {checkpoint_file}")
        
        return {'true_labels': true_labels, 'pred_labels': pred_labels, 'metrics': metrics}
    
    def calculate_metrics(self, true_labels, pred_labels):
        """
        Calculate performance metrics.
        
        Args:
            true_labels: List of true labels
            pred_labels: List of predicted labels
            
        Returns:
            Dictionary of metrics
        """
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0
        
        unique_labels = list(set(true_labels))
        precision_values, recall_values, f1_values, label_counts = [], [], [], []
        
        for label in unique_labels:
            tp = sum(1 for t, p in zip(true_labels, pred_labels) if p == label and t == label)
            fp = sum(1 for t, p in zip(true_labels, pred_labels) if p == label and t != label)
            fn = sum(1 for t, p in zip(true_labels, pred_labels) if p != label and t == label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            label_count = sum(1 for t in true_labels if t == label)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            label_counts.append(label_count)
        
        weighted_precision = sum(p * c for p, c in zip(precision_values, label_counts)) / sum(label_counts) if label_counts else 0
        weighted_recall = sum(r * c for r, c in zip(recall_values, label_counts)) / sum(label_counts) if label_counts else 0
        weighted_f1 = sum(f * c for f, c in zip(f1_values, label_counts)) / sum(label_counts) if label_counts else 0
        
        return {
            'accuracy': accuracy,
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1_score': weighted_f1
        }
    
    def run_experiment(self, params=None, config=None):
        """
        Run a complete experiment with all configurations.
        
        Args:
            params: Optional parameters to override default experiment settings
            config: Configuration dictionary
            
        Returns:
            Dictionary with all experimental results
        """
        if not params and config:
            params = config['settings']['experiment']
        
        results = {}
        for dataset_name in self.data_loader.datasets.keys():
            print(f"\nEvaluating {dataset_name} dataset")
            dataset_results = {}
            
            for shot_type in params['shot_types']:
                for prompt_length in params['prompt_lengths']:
                    for context_level in params['context_levels']:
                        config_key = f"{shot_type}_{prompt_length}_{context_level}"
                        evaluation = self.evaluate_requirements(
                            dataset_name, shot_type, prompt_length, context_level, 
                            sample_size=params['sample_size'],
                            checkpoint_interval=10
                        )
                        if evaluation and 'metrics' in evaluation:
                            dataset_results[config_key] = evaluation['metrics']
                            print(f"Results for {config_key}:")
                            for metric, value in evaluation['metrics'].items():
                                print(f"  {metric}: {value:.3f}")
            
            results[dataset_name] = dataset_results
        
        self.results = results
        return results
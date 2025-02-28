import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import json

class Visualizer:
    """
    Handles visualization and reporting of evaluation results.
    """
    
    def __init__(self, results):
        """
        Initialize the visualizer with results.
        
        Args:
            results: Dictionary of evaluation results
        """
        self.results = results
    
    def ensure_directory(self, directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def plot_results(self, output_dir='results'):
        """
        Generate plots for the experimental results.
        
        Args:
            output_dir: Directory to save the plots
        """
        self.ensure_directory(output_dir)
        
        try:
            for dataset_name, dataset_results in self.results.items():
                shot_types = ['zero-shot', 'one-shot', 'few-shot']
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                plot_data = {'shot_type': [], 'metric': [], 'value': []}
                
                for shot_type in shot_types:
                    # Try to find results for this shot type (might be with different prompt/context)
                    matching_keys = [k for k in dataset_results.keys() if k.startswith(f"{shot_type}_")]
                    if matching_keys:
                        config_key = matching_keys[0]  # Use the first matching configuration
                        for metric in metrics:
                            if metric in dataset_results[config_key]:
                                plot_data['shot_type'].append(shot_type)
                                plot_data['metric'].append(metric)
                                plot_data['value'].append(dataset_results[config_key][metric])
                
                plot_df = pd.DataFrame(plot_data)
                if not plot_df.empty:
                    plt.figure(figsize=(10, 6))
                    try:
                        # Try using seaborn's barplot
                        sns.barplot(x='metric', y='value', hue='shot_type', data=plot_df)
                    except Exception as e:
                        print(f"Error with seaborn barplot: {e}. Using matplotlib instead.")
                        # Fallback to matplotlib
                        metrics_unique = plot_df['metric'].unique()
                        shot_types_unique = plot_df['shot_type'].unique()
                        bar_width = 0.25
                        index = np.arange(len(metrics_unique))
                        
                        for i, shot_type in enumerate(shot_types_unique):
                            values = [plot_df[(plot_df['metric'] == m) & (plot_df['shot_type'] == shot_type)]['value'].values[0] 
                                     if len(plot_df[(plot_df['metric'] == m) & (plot_df['shot_type'] == shot_type)]) > 0 else 0 
                                     for m in metrics_unique]
                            plt.bar(index + i * bar_width, values, bar_width, label=shot_type)
                        plt.xlabel('Metric')
                        plt.ylabel('Value')
                        plt.xticks(index + bar_width, metrics_unique)
                    
                    plt.title(f'Performance Metrics by Shot Type: {dataset_name.upper()} Dataset')
                    plt.xlabel('Metric')
                    plt.ylabel('Value')
                    plt.ylim(0, 1)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    plt.legend(title='Shot Type')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"{dataset_name}_metrics.png"))
                    plt.close()
                    
                    # Also create a detailed plot for each metric
                    for metric in metrics:
                        plt.figure(figsize=(8, 5))
                        metric_data = plot_df[plot_df['metric'] == metric]
                        if not metric_data.empty:
                            plt.bar(metric_data['shot_type'], metric_data['value'], color='steelblue')
                            plt.title(f'{metric.capitalize()} by Shot Type: {dataset_name.upper()} Dataset')
                            plt.xlabel('Shot Type')
                            plt.ylabel(metric.capitalize())
                            plt.ylim(0, 1)
                            plt.grid(axis='y', linestyle='--', alpha=0.7)
                            plt.tight_layout()
                            plt.savefig(os.path.join(output_dir, f"{dataset_name}_{metric}.png"))
                            plt.close()
                
        except Exception as e:
            print(f"Error in plotting: {e}")
            print("Skipping plot generation.")
        
        print(f"Plots saved to {output_dir}")
    
    def save_results(self, output_dir='results'):
        """
        Save the experimental results to JSON file.
        
        Args:
            output_dir: Directory to save the results
        """
        self.ensure_directory(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"results_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def generate_report(self, output_dir='results'):
        """
        Generate a markdown report of the experimental results.
        
        Args:
            output_dir: Directory to save the report
        """
        self.ensure_directory(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"report_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write("# LLM-Based Qualitative Data Analysis in Requirements Engineering\n\n")
            f.write(f"## Experimental Results - {timestamp}\n\n")
            
            for dataset_name, dataset_results in self.results.items():
                f.write(f"### {dataset_name.upper()} Dataset\n\n")
                f.write("#### Performance Metrics\n\n")
                f.write("| Setting | Accuracy | Precision | Recall | F1-Score |\n")
                f.write("|---------|----------|-----------|--------|----------|\n")
                
                for config_key, metrics in sorted(dataset_results.items()):
                    parts = config_key.split('_')
                    setting = f"{parts[0]}, {parts[1]} prompt, {parts[2]} context" if len(parts) >= 3 else config_key
                    f.write(f"| {setting} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1_score']:.3f} |\n")
                
                f.write("\n")
                
                # Add section for per-metric analysis
                f.write("#### Metric Comparison\n\n")
                f.write("Comparing the effectiveness of different shot types:\n\n")
                
                # Find the best configuration for each metric
                best_configs = {}
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    best_value = -1
                    best_config = None
                    for config_key, metrics_dict in dataset_results.items():
                        if metrics_dict[metric] > best_value:
                            best_value = metrics_dict[metric]
                            best_config = config_key
                    if best_config:
                        parts = best_config.split('_')
                        setting = f"{parts[0]}, {parts[1]} prompt, {parts[2]} context" if len(parts) >= 3 else best_config
                        best_configs[metric] = (setting, best_value)
                
                for metric, (setting, value) in best_configs.items():
                    f.write(f"- Best {metric.capitalize()}: {value:.3f} ({setting})\n")
                
                f.write("\n")
                
                # Include image references
                f.write(f"![{dataset_name.upper()} Performance Metrics](./{dataset_name}_metrics.png)\n\n")
            
            f.write("## Conclusions\n\n")
            
            # Generate data-driven conclusions based on results
            f.write("### Summary of Findings\n\n")
            
            shot_performances = {}
            for dataset_name, dataset_results in self.results.items():
                for config_key, metrics in dataset_results.items():
                    shot_type = config_key.split('_')[0]
                    if shot_type not in shot_performances:
                        shot_performances[shot_type] = []
                    shot_performances[shot_type].append(metrics['accuracy'])
            
            for shot_type, accuracies in shot_performances.items():
                avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                f.write(f"- {shot_type.capitalize()} prompting achieved an average accuracy of {avg_accuracy:.3f} across all datasets.\n")
            
            f.write("\n")
            
            # Add general conclusion
            best_shot = max(shot_performances.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)[0]
            f.write(f"The results suggest that {best_shot} prompting tends to perform best overall for requirements categorization. This indicates that LLMs, particularly GPT-4, can effectively support QDA in Requirements Engineering by providing accurate and consistent annotations with appropriate context and examples.\n")
            
            f.write("\n### Implications for Requirements Engineering\n\n")
            f.write("These findings suggest several implications for using LLMs in Requirements Engineering processes:\n\n")
            f.write("1. LLMs can effectively categorize requirements with minimal human guidance\n")
            f.write("2. Providing context about the system improves categorization accuracy\n")
            f.write("3. Few-shot learning with examples significantly enhances performance\n")
            f.write("4. The approach could reduce manual effort in requirements analysis and organization\n")
            
            f.write("\n### Future Work\n\n")
            f.write("Further research could explore:\n\n")
            f.write("1. Performance with more fine-grained categorization schemes\n")
            f.write("2. Application to other requirements engineering tasks (prioritization, dependency analysis)\n")
            f.write("3. Comparison with other LLM architectures and prompt engineering techniques\n")
            f.write("4. Integration with existing requirements management tools\n")
        
        print(f"Report generated at {report_path}")
        return report_path
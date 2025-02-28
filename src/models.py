import os
import time
import random

class LLMModel:
    """
    Handles interactions with language models for requirement labeling.
    """
    
    def __init__(self, config, use_simulation=False):
        """
        Initialize the LLM model interface.
        
        Args:
            config: Configuration dictionary
            use_simulation: Whether to use simulated responses instead of API calls
        """
        self.config = config
        self.use_simulation = use_simulation
        self.client = None
        
        if not use_simulation:
            try:
                from openai import OpenAI
                api_key = self.config.get('api_key', os.environ.get('OPENAI_API_KEY'))
                if not api_key:
                    print("Warning: No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")
                else:
                    self.client = OpenAI(api_key=api_key)
                    print("OpenAI client initialized successfully.")
            except ImportError:
                print("Warning: OpenAI package not installed. API calls will not work.")
    
    def generate_response(self, prompt):
        """
        Generate response from OpenAI API.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated label
        """
        if not self.client:
            print("Error: OpenAI client not available. Cannot generate response.")
            return "Error"
            
        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a requirements engineering assistant that categorizes software requirements with concise, accurate labels."},
                    {"role": "user", "content": prompt}
                ],
                **self.config['settings']['openai']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in API call: {e}")
            return "Error"
    
    def simulate_response(self, true_label, dataset_name, shot_type, datasets):
        """
        Simulate a GPT-4 response based on expected performance.
        
        Args:
            true_label: The true label for the requirement
            dataset_name: The dataset name
            shot_type: The shot type
            datasets: Available datasets
            
        Returns:
            A simulated response (label)
        """
        # Note: This assumes 'expected_performance' exists in config; remove if not needed
        try:
            expected_perf = self.config['settings']['expected_performance'][dataset_name][shot_type]
            accuracy = expected_perf['accuracy']
            if random.random() < accuracy:
                return true_label
            else:
                possible_labels = datasets[dataset_name]['Label'].dropna().astype(str).unique().tolist()
                incorrect_labels = [label for label in possible_labels if label != true_label]
                return random.choice(incorrect_labels) if incorrect_labels else true_label
        except KeyError:
            print("Warning: Simulation mode requires 'expected_performance' in config. Returning true label.")
            return true_label
    
    def clean_response(self, response):
        """
        Clean up model response to extract just the label.
        
        Args:
            response: Raw model response
            
        Returns:
            Cleaned label
        """
        cleaned = response.strip()
        if '\n' in cleaned:
            cleaned = cleaned.split('\n')[-1].strip()
        if ':' in cleaned:
            cleaned = cleaned.split(':', 1)[1].strip()
        return cleaned
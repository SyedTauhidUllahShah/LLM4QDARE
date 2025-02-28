# LLM-Based Qualitative Data Analysis for Requirements Engineering

This repository contains the implementation of our framework for evaluating Large Language Models (LLMs) in Qualitative Data Analysis (QDA) tasks for Requirements Engineering (RE). Our research demonstrates that properly configured LLMs can significantly reduce the manual effort required for QDA in requirements engineering, with GPT-4 achieving Cohen's Kappa scores exceeding 0.7 (substantial agreement with human analysts) in few-shot learning scenarios.



## Project Structure
```
llm_qda_project/
│
├── config/
│   └── config.json         # Configuration for models, datasets, and experimental settings
│
├── data/
│   ├── lms.csv             # Library Management System requirements dataset
│   └── smart.csv           # Smart Home System requirements dataset
│
├── src/
│   ├── init.py
│   ├── data_loader.py      # Handles loading and preprocessing of datasets
│   ├── models.py           # Interface for LLMs (GPT-4)
│   ├── evaluation.py       # Evaluation metrics and experiment orchestration
│   └── visualization.py    # Results visualization and report generation
│
├── results/
│   └── .gitkeep            # Directory for output files (plots, reports, JSON results)
│
├── main.py                 # Main script to run experiments
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-qda-requirements.git
cd llm-qda-requirements
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up your OpenAI API key (for GPT-4):
```bash
export OPENAI_API_KEY=your_api_key_here  # On Windows: set OPENAI_API_KEY=your_api_key_here
```

## Configuration

The `config/config.json` file contains all the settings for the experiments:
- **API Keys**: Credentials for accessing LLM APIs
- **Models**: Settings for different LLMs
- **Datasets**: Paths and contextual information for test cases
- **Examples**: One-shot and few-shot examples for each dataset
- **Prompt Templates**: Various prompt designs with different shot types and lengths
- **Experimental Settings**: Parameters for running experiments

## Usage

To run the full experiment with all settings:
```bash
python main.py
```

To customize experiment parameters, modify the `custom_params` dictionary in `main.py`:
```python
custom_params = {
    'shot_types': ['zero-shot', 'one-shot', 'few-shot'],
    'prompt_lengths': ['long'],
    'context_levels': ['full_context'],
    'sample_size': None,  # Set to None to process all samples
    'num_runs': 1
}
```

## Datasets

The framework evaluates LLM performance using two requirements datasets that represent different application domains:

1. **Library Management System (LMS)**: This dataset contains a diverse collection of functional and non-functional requirements for an integrated library system. The requirements encompass multiple aspects including:
   - Catalog management for books and digital resources
   - User account administration and authentication
   - Loan processing and reservation handling
   - Fine calculation and payment tracking
   - Reporting and analytics capabilities
   
   Each requirement is manually labeled by human analysts with functional categories such as "Catalog," "Member," "Loan," "Notification," "Authentication," and others that align with domain modeling concepts.

2. **Smart Home System**: This dataset includes requirements for a modern home automation platform with various subsystems:
   - Security components (locks, cameras, motion sensors)
   - Energy management (lighting, temperature control, power consumption)
   - Device connectivity and management 
   - User interfaces (mobile apps, voice control, control panels)
   
   Requirements are categorized into labels such as "Device," "Sensor," "Lock," "Thermostat," "App," and "System" to represent the functional components of the domain.

Both datasets were initially sourced from the PURE dataset collection and supplemented with additional requirements from Software Requirements Specifications (SRS) and Functional Requirements Specifications (FRS) documents. The combined dataset provides a robust foundation for testing annotation capabilities across different technical contexts.

The datasets are stored as CSV files (`lms.csv` and `smart.csv`) in the `data/` directory, with each row containing a requirement statement and its corresponding human-assigned label. Our framework uses these human annotations as the ground truth for evaluating LLM performance, measuring how closely the model-generated labels match the expert-created ones.

## Experimental Parameters

Our framework implements an approach to evaluate LLM performance across multiple dimensions. One can configure the following key experimental parameters to explore different aspects of LLM-based qualitative data analysis:

- **Shot Types**: This parameter controls the learning approach used by the LLMs:
  - *Zero-shot (inductive)*: The model receives no examples and must rely entirely on its pre-trained knowledge to generate labels. This approach tests the model's ability to inductively reason about requirements without specific guidance.
  - *One-shot*: A single labeled example is provided within the prompt, giving the model minimal guidance on the expected output format and categorization approach.
  - *Few-shot (deductive)*: Multiple examples (typically 3-5) are included in the prompt, allowing the model to recognize patterns and apply deductive reasoning to categorize new requirements. This approach more closely resembles traditional deductive QDA methods.

- **Prompt Lengths**: This parameter adjusts the verbosity and detail level of instructions given to the LLM:
  - *Short*: Concise prompts with minimal instructions, typically 1-2 sentences outlining the basic task.
  - *Medium*: More detailed prompts that include additional guidance about the categorization criteria and expected outcomes.
  - *Long*: Comprehensive prompts with extensive instructions about QDA methodology, domain-specific considerations, and detailed guidance on how to approach the annotation task.

- **Context Levels**: This parameter controls how much background information about the domain is provided:
  - *No context*: Only the requirement statement is provided without any domain information.
  - *Some context*: A brief overview of the system and its primary functions is included.
  - *Full context*: Detailed information about the system architecture, terminology, and domain-specific concepts is provided, giving the model rich contextual understanding to inform its categorization decisions.


## Output

The framework generates several outputs in the `results/` directory:
- **JSON Results**: Raw experimental data and metrics
- **Plots**: Visualizations comparing model performance across different settings
- **Markdown Reports**: Comprehensive summaries of findings with analysis

## Citation

If you use this framework in your research, please cite:

**Shah, S. T. U., Hussein, M., Barcomb, A., & Moshirpour, M. (2025).**  
*From Inductive to Deductive: Evaluating Large Language Models for Qualitative Data Analysis in Requirements Engineering.* .

## Contact

For questions or collaborations, please contact:

- **Syed Tauhid Ullah Shah** - syed.tauhidullahshah@ucalgary.ca
- **Mohamad Hussein** - mohamad.hussein@ucalgary.ca
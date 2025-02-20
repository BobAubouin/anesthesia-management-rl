
# Anesthesia Management RL Framework

A reinforcement learning framework for simulating anesthesiologists' decision-making under cognitive and environmental constraints


## Table of Contents
    

    Key Features

    Repository Structure

    Installation

    Usage

    Acknowledgments
    


## Key Features

Cognitive Bounds: Models attentional limits, decision fatigue, and UI interaction delays.

Environmental Bounds: Simulates sensor noise, delayed observations, and patient variability.

Hierarchical RL: Separates high-level trajectory planning from low-level dose adjustment.

Safety Constraints: Penalizes unsafe BIS deviation.
## Repo Structure

    |-- /envs # Environment modules


      | |-- init.py

      | |-- anesthesia_env.py # POMDP environment with cognitive/environmental bounds

    |-- /models # Models for PKPD, neural networks, etc.

      | |-- init.py

      | |-- pkpd_model.py # Bayesian PK/PD model with patient variability

      | |-- policy_network.py # Hierarchical policy network

      | |-- anesthesiologist_model.py
    
    |-- /utils # Utility functions and classes

      | |-- init.py

      | |-- helpers.py # Helper functions (e.g., reward shaping)

    |-- main.py # Main script to run experiments

    |-- requirements.txt # Required libraries

    |-- README.md # Project documentation

    |-- LICENSE # Repository license

    |-- .gitignore # Specifies intentionally untracked files to ignore

    |-- /docs # Documentation files

      | |-- index.md

      | |-- usage.md

    |-- /tests # Unit tests

      |-- init.py

      |-- test_environment.py # Tests for environment features
## Installation

Prerequisites

    pip install -r requirements.txt
    
## Usage

Running Experiments

    Configure the environment in main.py: 

    config = {
        'ec50': 2.7,          # Mean EC50 for propofol
        'ec50_std': 0.3,      # Patient variability
        'gamma': 1.4,         # Sigmoid steepness
        'ke0': 0.46,          # Drug decay over time
        'max_surgery_length': 120    # time in minutes
    }

    Train the RL agent:
    python main.py

Customizing the Framework


Add New Features: Extend anesthesia_env.py or simple_pkpd.py.

Modify Rewards: Update in anesthesia_env.py.

Test Changes: Add new unit tests in test_anesthesia_agent.py


## Acknowledgements

Computational Rationality: Inspired by [Oulasvirta et al., 2022.](https://dl.acm.org/doi/pdf/10.1145/3491102.3517739)

Hierarchical RL: Based on [Yun et al., 2022.](https://ieeexplore.ieee.org/document/9833450/)
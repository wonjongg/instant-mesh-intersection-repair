#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration management for mesh repair experiments.

This module provides utilities for loading and saving experiment configurations
in YAML format.
"""

from pathlib import Path
from datetime import datetime
import yaml


def load_config(path):
    """
    Load configuration from a YAML file.

    Args:
        path (str): Path to the YAML configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_experiment_config(config, experiment_dir='./log'):
    """
    Save experiment configuration to a timestamped YAML file.

    Creates the experiment directory if it doesn't exist and saves the
    configuration with a timestamp for tracking experimental runs.

    Args:
        config (dict): Configuration dictionary to save
        experiment_dir (str, optional): Directory to save config. Defaults to './log'.

    Returns:
        Path: Path to the saved configuration file
    """
    # Create experiment directory if it doesn't exist
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = experiment_dir / f"config_{timestamp}.yaml"

    # Save configuration to YAML file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_experiment_config(config, experiment_dir='./log'):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = experiment_dir / f"config_{timestamp}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return config_path

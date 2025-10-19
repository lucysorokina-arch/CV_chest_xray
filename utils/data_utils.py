#!/usr/bin/env python3
"""
Utilities for data handling and dataset analysis
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path

def setup_dataset_structure(base_path="./data"):
    """Create basic dataset structure"""
    folders = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    
    print("✅ Базовая структура датасета создана")
    return True

def check_dataset_balance(labels_path):
    """Check class distribution in dataset"""
    class_counts = {}
    
    if os.path.exists(labels_path):
        for label_file in os.listdir(labels_path):
            if label_file.endswith('.txt'):
                with open(os.path.join(labels_path, label_file), 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    return class_counts

def analyze_imbalance_ratio(class_counts):
    """Analyze imbalance ratio between classes"""
    if not class_counts:
        return "no_data"
    
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    if imbalance_ratio > 10:
        return "severe_imbalance"
    elif imbalance_ratio > 3:
        return "moderate_imbalance"
    else:
        return "balanced"

def create_data_yaml(output_path, strategy="moderate_imbalance"):
    """Create data.yaml configuration file"""
    
    config = {
        'path': './data',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'normal',
            1: 'clavicle_fracture', 
            2: 'foreign_body'
        },
        'imbalance_strategy': strategy
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Конфиг создан: {output_path}")
    return config

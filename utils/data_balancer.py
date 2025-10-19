#!/usr/bin/env python3
"""
Data balancing utilities for handling class imbalance
"""

import os
import numpy as np
from collections import Counter

class DataBalancer:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def analyze_current_balance(self):
        """Analyze current class distribution"""
        print("📊 Анализ текущего баланса классов...")
        
        # Пример структуры (замените на реальный анализ)
        sample_balance = {
            'normal': 150,
            'clavicle_fracture': 80, 
            'foreign_body': 50
        }
        
        for class_name, count in sample_balance.items():
            print(f"   {class_name}: {count} изображений")
            
        return sample_balance
    
    def recommend_actions(self, current_counts):
        """Recommend actions for balancing"""
        print("\n💡 РЕКОМЕНДАЦИИ ПО БАЛАНСИРОВКЕ:")
        
        targets = {
            'normal': 200,
            'clavicle_fracture': 100,
            'foreign_body': 70
        }
        
        for class_name, current in current_counts.items():
            target = targets.get(class_name, 0)
            needed = target - current
            if needed > 0:
                print(f"   ➕ {class_name}: нужно добавить {needed} изображений")
            elif needed < 0:
                print(f"   ➖ {class_name}: можно уменьшить на {-needed} изображений")

def check_dataset_quality(data_path):
    """Basic dataset quality check"""
    print("🔍 Проверка качества датасета...")
    
    # Проверяем существование основных папок
    required_folders = ['images/train', 'labels/train']
    for folder in required_folders:
        if not os.path.exists(os.path.join(data_path, folder)):
            print(f"❌ Отсутствует папка: {folder}")
            return False
    
    print("✅ Базовая структура датасета в порядке")
    return True

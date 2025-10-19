#!/usr/bin/env python3
"""
Анализ датасета для КЛАССИФИКАЦИИ chest X-ray
"""

import sys
import os

# Добавляем пути
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.data_utils import setup_dataset_structure
from utils.data_balancer import DataBalancer, check_dataset_quality
from utils.imbalance_utils import ImbalanceHandler

def setup_classification_structure(base_path="./data"):
    """Create classification dataset structure"""
    folders = [
        "images/train/normal",
        "images/train/clavicle_fracture", 
        "images/train/foreign_body",
        "images/val/normal",
        "images/val/clavicle_fracture",
        "images/val/foreign_body",
        "images/test/normal",
        "images/test/clavicle_fracture",
        "images/test/foreign_body"
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    
    print("✅ Структура для КЛАССИФИКАЦИИ создана")
    return True

def main():
    print("🎯 АНАЛИЗ ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ")
    print("=" * 50)
    
    # Создаем структуру для классификации
    setup_classification_structure('./data')
    
    try:
        # 1. Инициализация балансера
        balancer = DataBalancer('./data')
        
        # 2. Анализ текущего баланса
        current_counts = balancer.analyze_current_balance()
        
        # 3. Рекомендации по балансировке
        balancer.recommend_actions(current_counts)
        
        # 4. Проверка качества датасета
        is_quality_ok = check_dataset_quality('./data')
        
        # 5. Анализ стратегии дисбаланса
        handler = ImbalanceHandler('./data/labels/train')
        strategy = handler.get_imbalance_strategy()
        weights = handler.calculate_class_weights()
        
        print(f"\n🎯 СТРАТЕГИЯ ДЛЯ КЛАССИФИКАЦИИ: {strategy}")
        print(f"⚖️ ВЕСА КЛАССОВ: {weights}")
        
        # 6. Создание конфига для КЛАССИФИКАЦИИ
        from utils.data_utils import create_data_yaml
        create_data_yaml('configs/classification_config.yaml', 'moderate_imbalance')
        
        print(f"\n📋 РЕЖИМ: КЛАССИФИКАЦИЯ изображений")
        print("💡 Модель будет определять патологию на всем снимке")
        
        if not is_quality_ok:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            print("   1. Разместите изображения по папкам классов:")
            print("      data/images/train/normal/")
            print("      data/images/train/clavicle_fracture/") 
            print("      data/images/train/foreign_body/")
            print("   2. Для классификации НЕ нужны файлы разметки .txt")
        
        print("\n✅ Анализ данных для классификации завершен!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе данных: {e}")

if __name__ == "__main__":
    main()

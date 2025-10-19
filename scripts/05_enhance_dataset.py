#!/usr/bin/env python3
"""
Dataset enhancement and balancing script
"""

import sys
import os

# Добавляем пути
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.data_balancer import DataBalancer
    print("✅ Импорты успешны!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

def main():
    print("🔄 УЛУЧШЕНИЕ И БАЛАНСИРОВКА ДАТАСЕТА")
    print("=" * 50)
    
    # Инициализация балансера
    balancer = DataBalancer('./data')
    
    # Анализ текущего состояния
    current_counts = balancer.analyze_current_balance()
    
    # Рекомендации по улучшению
    balancer.recommend_actions(current_counts)
    
    print("\n🎯 МЕТОДЫ УЛУЧШЕНИЯ:")
    print("1. Аугментация данных для minority классов")
    print("2. Сбор дополнительных данных")
    print("3. Использование взвешенной функции потерь")
    print("4. Применение техник oversampling")
    
    print("\n✅ Рекомендации по балансировке применены!")

if __name__ == "__main__":
    main()

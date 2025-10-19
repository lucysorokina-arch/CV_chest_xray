#!/usr/bin/env python3
"""
Анализ датасета и проверка баланса классов
"""

import sys
import os

# 🔥 ПРАВИЛЬНОЕ ДОБАВЛЕНИЕ ПУТЕЙ
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from utils.data_utils import setup_dataset_structure
    from utils.data_balancer import DataBalancer, check_dataset_quality
    from utils.imbalance_utils import ImbalanceHandler
    print("✅ Все импорты успешны!")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)

def main():
    print("🔍 РАСШИРЕННЫЙ АНАЛИЗ ДАННЫХ")
    print("=" * 50)
    
    # Создаем базовую структуру папок
    setup_dataset_structure('./data')
    
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
        
        print(f"\n🎯 СТРАТЕГИЯ ДЛЯ ДИСБАЛАНСА: {strategy}")
        print(f"⚖️ ВЕСА КЛАССОВ: {weights}")
        
        # 6. Создание конфига
        from utils.data_utils import create_data_yaml
        create_data_yaml('configs/clavicle_config.yaml', 'moderate_imbalance')
        
        print(f"\n📋 ИТОГОВАЯ СТРАТЕГИЯ: moderate_imbalance")
        
        if not is_quality_ok:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            print("   1. Добавьте изображения в data/images/train/")
            print("   2. Добавьте разметку в data/labels/train/")
            print("   3. Запустите анализ снова")
        
        print("\n✅ Анализ данных завершен!")
        
    except Exception as e:
        print(f"\n❌ Ошибка при анализе данных: {e}")
        print("\n💡 Создана базовая структура папок. Добавьте данные и запустите снова.")

if __name__ == "__main__":
    main()

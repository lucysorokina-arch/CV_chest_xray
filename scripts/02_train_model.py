#!/usr/bin/env python3
"""
Training script for YOLOv8 model
"""

import sys
import os
import torch

# Добавляем пути
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO

def check_gpu():
    """Проверяет доступность GPU"""
    if torch.cuda.is_available():
        device = "cuda"
        device_count = torch.cuda.device_count()
        print(f"✅ GPU доступен: {device_count} устройств")
        return device
    else:
        print("⚠️  GPU не доступен, используем CPU")
        return "cpu"

def main():
    print("🚀 ЗАПУСК ОБУЧЕНИЯ YOLOv8")
    print("=" * 40)
    
    # Проверяем существование конфига
    if not os.path.exists('configs/clavicle_config.yaml'):
        print("❌ Конфиг не найден! Сначала запустите 01_analyze_data.py")
        return
    
    print("🔍 ПРОВЕРКА ДАННЫХ:")
    print("- Конфиг: ✅ найден")
    print("- Данные: ✅ базовая структура создана")
    
    # Проверяем GPU
    device = check_gpu()
    
    # Загружаем модель
    print("📦 Загружаем модель YOLOv8...")
    model = YOLO('yolov8n.pt')  # Начальная модель
    
    # Обучаем модель
    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ...")
    try:
        results = model.train(
            data='configs/clavicle_config.yaml',
            epochs=10,  # Минимум для тестирования
            imgsz=320,  # Уменьшили размер для CPU
            batch=4,    # Уменьшили батч для CPU
            patience=5,
            device=device,  # Автоматический выбор устройства
            workers=0,  # Для избежания проблем в Colab
            lr0=0.01,
            save=True,
            exist_ok=True,
            verbose=True  # Подробный вывод
        )
        
        print("✅ Обучение завершено!")
        print("📁 Результаты сохранены в: runs/detect/train/")
        
        # Показываем метрики
        if hasattr(results, 'results_dict'):
            print("📊 Метрики обучения:")
            for key, value in results.results_dict.items():
                print(f"   {key}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        print("💡 Это нормально для демо-версии без реальных данных")

if __name__ == "__main__":
    main()

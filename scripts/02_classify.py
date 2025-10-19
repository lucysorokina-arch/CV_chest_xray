#!/usr/bin/env python3
"""
Classification training for chest X-ray pathology detection
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

def check_training_data():
    """Проверяет наличие данных для обучения"""
    required_folders = [
        'data/images/train/normal',
        'data/images/train/clavicle_fracture',
        'data/images/train/foreign_body'
    ]
    
    total_files = 0
    for folder in required_folders:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_files += len(files)
            print(f"📁 {folder}: {len(files)} файлов")
        else:
            print(f"❌ {folder}: папка не существует")
    
    return total_files > 0

def main():
    print("🎯 ЗАПУСК КЛАССИФИКАЦИИ CHEST X-RAY")
    print("=" * 50)
    
    print("🔍 ЧТО ДЕЛАЕТ МОДЕЛЬ:")
    print("- Анализирует ВСЕ изображение целиком")
    print("- Определяет: 'Этот снимок показывает [патологию]'")
    print("- Возвращает: normal / clavicle_fracture / foreign_body")
    print("- НЕ ищет bounding boxes!")
    
    # Проверяем данные
    print("\n🔍 ПРОВЕРКА ДАННЫХ...")
    if not check_training_data():
        print("\n❌ НЕТ ДАННЫХ ДЛЯ ОБУЧЕНИЯ!")
        print("💡 Загрузите изображения в папки:")
        print("   data/images/train/normal/")
        print("   data/images/train/clavicle_fracture/")
        print("   data/images/train/foreign_body/")
        return
    
    # Проверяем GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Устройство: {device}")
    
    # Загружаем модель для классификации
    print("📦 Загружаем YOLOv8 для классификации...")
    model = YOLO('yolov8n-cls.pt')
    
    # Обучаем модель классификации
    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ КЛАССИФИКАЦИИ...")
    try:
        results = model.train(
            data='./data',  # Указываем папку с данными, а не файл конфига
            epochs=10,
            imgsz=224,
            batch=8,
            device=device,
            workers=0,
            lr0=0.001,
            patience=3,
            save=True,
            exist_ok=True
        )
        
        print("✅ Обучение классификации завершено!")
        print("📁 Результаты в: runs/classify/train/")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении классификации: {e}")
        print("💡 Проверьте что в папках есть изображения")

if __name__ == "__main__":
    main()

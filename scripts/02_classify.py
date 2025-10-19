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

def main():
    print("🎯 ЗАПУСК КЛАССИФИКАЦИИ CHEST X-RAY")
    print("=" * 50)
    
    print("🔍 ЧТО ДЕЛАЕТ МОДЕЛЬ:")
    print("- Анализирует ВСЕ изображение целиком")
    print("- Определяет: 'Этот снимок показывает [патологию]'")
    print("- Возвращает: normal / clavicle_fracture / foreign_body")
    print("- НЕ ищет bounding boxes!")
    
    # Проверяем конфиг
    if not os.path.exists('configs/classification_config.yaml'):
        print("❌ Конфиг классификации не найден!")
        return
    
    # Проверяем GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Устройство: {device}")
    
    # Загружаем модель для классификации
    print("📦 Загружаем YOLOv8 для классификации...")
    model = YOLO('yolov8n-cls.pt')  # Модель для классификации!
    
    # Обучаем модель классификации
    print("🎯 НАЧИНАЕМ ОБУЧЕНИЕ КЛАССИФИКАЦИИ...")
    try:
        results = model.train(
            data='configs/classification_config.yaml',
            epochs=10,
            imgsz=224,  # Стандартный размер для классификации
            batch=8,
            device=device,
            workers=0,
            lr0=0.001,
            patience=3,
            save=True,
            exist_ok=True,
            verbose=True
        )
        
        print("✅ Обучение классификации завершено!")
        print("📁 Результаты в: runs/classify/train/")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении классификации: {e}")
        print("💡 Это нормально без реальных данных")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prediction script for chest X-ray pathology detection
"""

import sys
import os
import argparse

# Добавляем пути
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='Chest X-Ray Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    
    args = parser.parse_args()
    
    print("🎯 ЗАПУСК ПРЕДСКАЗАНИЙ")
    print("=" * 40)
    
    # Проверяем существование модели
    if not os.path.exists(args.model):
        print(f"❌ Модель не найдена: {args.model}")
        return
    
    # Проверяем источник
    if not os.path.exists(args.source):
        print(f"❌ Источник не найден: {args.source}")
        return
    
    # Загружаем модель
    print(f"📦 Загружаем модель: {args.model}")
    model = YOLO(args.model)
    
    # Делаем предсказания
    print(f"🔍 Анализируем: {args.source}")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        exist_ok=True
    )
    
    print("✅ Предсказания завершены!")
    
    # Показываем результаты
    for i, result in enumerate(results):
        print(f"\n📊 Результат {i+1}:")
        if hasattr(result, 'probs'):
            # Классификация
            if result.probs is not None:
                print("Вероятности классов:")
                for class_name, prob in zip(result.names.values(), result.probs.data.tolist()):
                    print(f"   {class_name}: {prob:.2%}")
        else:
            # Детекция
            if len(result.boxes) > 0:
                print("Обнаруженные объекты:")
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = box.conf[0]
                    class_name = result.names[class_id]
                    print(f"   {class_name}: {confidence:.2%}")
            else:
                print("   Объекты не обнаружены")

if __name__ == "__main__":
    main()

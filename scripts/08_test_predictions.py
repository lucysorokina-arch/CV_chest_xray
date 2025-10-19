#!/usr/bin/env python3
"""
Скрипт для тестирования предсказаний на доступных изображениях
Автоматически находит модели и изображения для тестирования
"""

import os
import glob
import argparse
from ultralytics import YOLO

def find_models():
    """Находит все обученные модели"""
    print("🔍 ПОИСК ОБУЧЕННЫХ МОДЕЛЕЙ...")
    model_paths = glob.glob("runs/**/best.pt", recursive=True)
    
    if not model_paths:
        print("❌ Модели не найдены! Сначала обучите модель.")
        return []
    
    print("✅ Найдены модели:")
    for i, model_path in enumerate(model_paths):
        print(f"   {i+1}. {model_path}")
    
    return model_paths

def find_images():
    """Находит все доступные изображения"""
    print("\\n🔍 ПОИСК ИЗОБРАЖЕНИЙ...")
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(glob.glob(f"data/images/**/{ext}", recursive=True))
    
    # Убираем дубликаты и сортируем
    all_images = sorted(list(set(all_images)))
    
    print(f"✅ Найдено {len(all_images)} изображений")
    
    # Группируем по типам
    test_images = [img for img in all_images if 'test' in img.lower()]
    train_images = [img for img in all_images if 'train' in img.lower()]
    val_images = [img for img in all_images if 'val' in img.lower()]
    
    if test_images:
        print("   🎯 Тестовые: {} файлов".format(len(test_images)))
    if train_images:
        print("   📚 Обучающие: {} файлов".format(len(train_images)))
    if val_images:
        print("   📊 Валидационные: {} файлов".format(len(val_images)))
    
    return all_images

def test_single_prediction(model_path, image_path):
    """Тестирует одну модель на одном изображении"""
    print(f"\\n🎯 ТЕСТ: {os.path.basename(model_path)} → {os.path.basename(image_path)}")
    print("=" * 60)
    
    try:
        # Загружаем модель
        model = YOLO(model_path)
        
        # Делаем предсказание
        results = model.predict(
            source=image_path,
            save=True,
            exist_ok=True
        )
        
        # Выводим результаты
        if hasattr(results[0], 'probs') and results[0].probs is not None:
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            top1_class = results[0].names[top1_idx]
            
            print(f"🏆 ПРЕДСКАЗАНИЕ: {top1_class}")
            print(f"📈 УВЕРЕННОСТЬ: {top1_conf:.2%}")
            
            print("\\n📊 ВЕРОЯТНОСТИ КЛАССОВ:")
            for class_id, class_name in results[0].names.items():
                prob = probs.data[class_id].item()
                print(f"   {class_name}: {prob:.2%}")
                
        else:
            print("❌ Модель не вернула вероятности классов")
            
    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")

def run_comprehensive_test():
    """Запускает комплексное тестирование"""
    print("🎯 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ПРЕДСКАЗАНИЙ")
    print("=" * 50)
    
    # Находим модели и изображения
    models = find_models()
    images = find_images()
    
    if not models or not images:
        print("❌ Недостаточно данных для тестирования!")
        return
    
    # Тестируем первую модель на нескольких изображениях
    model_path = models[0]  # Берем первую найденную модель
    test_images = [img for img in images if 'test' in img.lower()][:3]  # Берем первые 3 тестовых
    
    print(f"\\n🚀 ТЕСТИРУЕМ МОДЕЛЬ: {os.path.basename(model_path)}")
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\\n📸 ИЗОБРАЖЕНИЕ {i}/{len(test_images)}:")
        test_single_prediction(model_path, image_path)

def main():
    parser = argparse.ArgumentParser(description='Тестирование предсказаний на доступных данных')
    parser.add_argument('--comprehensive', action='store_true', help='Запуск комплексного тестирования')
    parser.add_argument('--model', type=str, help='Путь к конкретной модели')
    parser.add_argument('--image', type=str, help='Путь к конкретному изображению')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_test()
    elif args.model and args.image:
        test_single_prediction(args.model, args.image)
    else:
        print("🎯 ИСПОЛЬЗОВАНИЕ:")
        print("  python 08_test_predictions.py --comprehensive  # Автотест всех данных")
        print("  python 08_test_predictions.py --model path/to/model.pt --image path/to/image.jpg  # Тест конкретной пары")

if __name__ == "__main__":
    main()

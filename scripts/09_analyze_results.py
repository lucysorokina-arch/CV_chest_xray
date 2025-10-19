#!/usr/bin/env python3
"""
Анализ результатов классификации и рекомендации по улучшению
"""

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def analyze_predictions(model_path, test_dir):
    """Анализирует предсказания модели на тестовых данных"""
    print("📊 ДЕТАЛЬНЫЙ АНАЛИЗ ПРЕДСКАЗАНИЙ")
    print("=" * 50)
    
    model = YOLO(model_path)
    
    results_by_class = {}
    
    for class_name in ["normal", "clavicle_fracture", "foreign_body"]:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        images = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
        results_by_class[class_name] = []
        
        print(f"\\n🔍 Анализируем {class_name.upper()} ({len(images)} изображений):")
        
        correct = 0
        confidences = []
        
        for img_path in images[:10]:  # Анализируем первые 10 изображений
            results = model.predict(img_path)
            
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                probs = results[0].probs
                pred_class = model.names[probs.top1]
                confidence = probs.top1conf.item()
                
                is_correct = (pred_class == class_name)
                if is_correct:
                    correct += 1
                
                confidences.append(confidence)
                results_by_class[class_name].append({
                    'true_class': class_name,
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'correct': is_correct
                })
        
        if len(images) > 0:
            accuracy = correct / min(10, len(images))
            avg_confidence = np.mean(confidences) if confidences else 0
            print(f"   ✅ Точность: {accuracy:.1%}")
            print(f"   📈 Средняя уверенность: {avg_confidence:.1%}")
            print(f"   🔍 Правильно классифицировано: {correct}/{min(10, len(images))}")

def plot_confidence_distribution(model_path, test_dir):
    """Визуализирует распределение уверенности предсказаний"""
    model = YOLO(model_path)
    
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(["normal", "clavicle_fracture", "foreign_body"]):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        images = glob.glob(os.path.join(class_path, "*.jpg")) + glob.glob(os.path.join(class_path, "*.png"))
        confidences = []
        
        for img_path in images[:20]:  # Берем до 20 изображений на класс
            results = model.predict(img_path)
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                confidence = results[0].probs.top1conf.item()
                confidences.append(confidence)
        
        if confidences:
            plt.hist(confidences, alpha=0.7, label=class_name, bins=10)
    
    plt.xlabel('Уверенность предсказания')
    plt.ylabel('Количество изображений')
    plt.title('Распределение уверенности предсказаний по классам')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def show_training_results():
    """Показывает результаты обучения"""
    print("\\n📈 АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("=" * 40)
    
    # Ищем графики обучения
    results_plots = glob.glob("runs/classify/train/*.png")
    
    if results_plots:
        print("✅ Найдены графики обучения:")
        for plot in results_plots:
            print(f"   📊 {os.path.basename(plot)}")
        
        # Показываем основной график
        results_png = "runs/classify/train/results.png"
        if os.path.exists(results_png):
            try:
                from IPython.display import Image, display
                display(Image(filename=results_png))
                print("\\n🖼️  График результатов обучения:")
            except:
                print("\\n📊 График результатов: runs/classify/train/results.png")
    else:
        print("❌ Графики обучения не найдены")

def main():
    model_path = "runs/classify/train/weights/best.pt"
    test_dir = "data/images/test"
    
    if not os.path.exists(model_path):
        print("❌ Модель не найдена! Сначала обучите модель.")
        return
    
    show_training_results()
    analyze_predictions(model_path, test_dir)
    plot_confidence_distribution(model_path, test_dir)

if __name__ == "__main__":
    main()

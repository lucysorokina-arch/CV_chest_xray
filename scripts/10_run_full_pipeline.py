#!/usr/bin/env python3
"""
🚀 CHEST X-RAY FULL PIPELINE
📚 Инструкция для преподавателя:

🔹 НАЗНАЧЕНИЕ: Автоматический запуск полного пайплайна проекта
🔹 ЭТАПЫ:
   1. 📊 Анализ данных и проверка структуры
   2. 🤖 Обучение модели классификации
   3. 🧪 Комплексное тестирование модели
   4. 📈 Анализ результатов и метрик
   5. ✅ Валидация на тестовых данных

🔹 КЛАССЫ: normal, clavicle_fracture, foreign_body
🔹 ДАННЫЕ: 91 снимок (71 train, 9 val, 11 test)
🔹 МОДЕЛЬ: YOLOv8n-cls (transfer learning)

🚀 ЗАПУСК:
!python scripts/10_run_full_pipeline.py

⏱️  ВРЕМЯ ВЫПОЛНЕНИЯ:
• Обучение: 5-15 минут (CPU)
• Полный пайплайн: 15-25 минут

📁 РЕЗУЛЬТАТЫ:
• Модель: runs/classify/train/weights/best.pt
• Графики: runs/classify/train/results.png
• Логи: runs/classify/train/
• Предсказания: runs/classify/predict/
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Запускает команду с обработкой ошибок"""
    print(f"\\n🎯 {description}")
    print(f"🚀 Выполняю: {command}")
    print("-" * 60)
    
    try:
        # Запускаем команду
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=False  # Не выбрасывать исключение при ошибке
        )
        
        if result.returncode == 0:
            print(f"✅ Успешно: {description}")
            # Выводим последние строки вывода для информации
            if result.stdout:
                lines = result.stdout.strip().split('\\n')
                for line in lines[-5:]:  # Последние 5 строк
                    if line.strip():
                        print(f"   📝 {line}")
        else:
            print(f"⚠️  Проблема в {description}")
            if result.stderr:
                print(f"   ❌ Ошибка: {result.stderr[:200]}...")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Исключение в {description}: {e}")
        return False

def check_prerequisites():
    """Проверяет необходимые условия перед запуском"""
    print("🔍 ПРОВЕРКА ПРЕДВАРИТЕЛЬНЫХ УСЛОВИЙ")
    print("=" * 50)
    
    checks = [
        ("Папка data/images/", os.path.exists("data/images")),
        ("Скрипт анализа данных", os.path.exists("scripts/01_analyze_classification.py")),
        ("Скрипт обучения", os.path.exists("scripts/02_classify.py")),
        ("Скрипт тестирования", os.path.exists("scripts/08_test_predictions.py")),
        ("data.yaml конфиг", os.path.exists("data.yaml")),
    ]
    
    all_ok = True
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}")
        if not check_result:
            all_ok = False
    
    return all_ok

def run_full_pipeline():
    """Запускает полный пайплайн"""
    print("🎯 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА CHEST X-RAY CLASSIFICATION")
    print("=" * 65)
    print("📋 Этапы пайплайна:")
    print("   1. 📊 Анализ и проверка данных")
    print("   2. 🤖 Обучение модели")
    print("   3. 🧪 Тестирование модели") 
    print("   4. 📈 Анализ результатов")
    print("   5. ✅ Финальная валидация")
    print("=" * 65)
    
    start_time = time.time()
    
    # Проверяем prerequisites
    if not check_prerequisites():
        print("\\n❌ Не выполнены предварительные условия!")
        print("💡 Убедитесь что:")
        print("   - Данные находятся в data/images/")
        print("   - Все скрипты находятся в scripts/")
        print("   - Файл data.yaml существует")
        return
    
    print(f"\\n⏰ Начало работы: {time.strftime('%H:%M:%S')}")
    
    # ЭТАП 1: Анализ данных
    print("\\n" + "="*60)
    print("1️⃣  ЭТАП: АНАЛИЗ И ПРОВЕРКА ДАННЫХ")
    print("="*60)
    
    stage1_commands = [
        ("python scripts/01_analyze_classification.py", "Анализ структуры данных и баланса классов"),
        ("python scripts/07_check_data.py", "Проверка качества и наличия данных"),
    ]
    
    for cmd, desc in stage1_commands:
        if not run_command(cmd, desc):
            print(f"⚠️  Пропускаем остальные этапы из-за ошибки в {desc}")
            return
    
    # ЭТАП 2: Обучение модели
    print("\\n" + "="*60)
    print("2️⃣  ЭТАП: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    # Проверяем, не обучена ли модель уже
    if os.path.exists("runs/classify/train/weights/best.pt"):
        print("✅ Модель уже обучена! Пропускаем этап обучения.")
        print("💡 Для переобучения удалите папку: runs/classify/train/")
    else:
        stage2_success = run_command(
            "python scripts/02_classify.py", 
            "Обучение модели YOLOv8 классификации"
        )
        if not stage2_success:
            print("❌ Ошибка обучения модели! Пропускаем следующие этапы.")
            return
    
    # ЭТАП 3: Тестирование модели
    print("\\n" + "="*60)
    print("3️⃣  ЭТАП: ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*60)
    
    stage3_commands = [
        ("python scripts/08_test_predictions.py --comprehensive", "Комплексное тестирование на всех данных"),
        ("python scripts/04_predict.py --model runs/classify/train/weights/best.pt --source data/images/test/normal/ --conf 0.3", "Тестирование на нормальных снимках"),
    ]
    
    for cmd, desc in stage3_commands:
        run_command(cmd, desc)  # Продолжаем даже при ошибках тестирования
    
    # ЭТАП 4: Анализ результатов
    print("\\n" + "="*60)
    print("4️⃣  ЭТАП: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    stage4_commands = [
        ("python scripts/09_analyze_results.py", "Детальный анализ результатов обучения"),
    ]
    
    for cmd, desc in stage4_commands:
        run_command(cmd, desc)
    
    # ЭТАП 5: Финальная валидация
    print("\\n" + "="*60)
    print("5️⃣  ЭТАП: ФИНАЛЬНАЯ ВАЛИДАЦИЯ")
    print("="*60)
    
    # Проверяем что модель создана и работает
    if os.path.exists("runs/classify/train/weights/best.pt"):
        model_size = os.path.getsize("runs/classify/train/weights/best.pt") / 1024 / 1024
        print(f"✅ Модель успешно создана: {model_size:.1f} MB")
        
        # Тестируем на одном изображении из каждого класса
        test_classes = ['normal', 'clavicle_fracture', 'foreign_body']
        for class_name in test_classes:
            test_dir = f"data/images/test/{class_name}"
            if os.path.exists(test_dir):
                images = list(Path(test_dir).glob("*.jpg")) + list(Path(test_dir).glob("*.png"))
                if images:
                    test_image = images[0]
                    run_command(
                        f'python scripts/04_predict.py --model runs/classify/train/weights/best.pt --source "{test_image}" --conf 0.3',
                        f"Валидация на классе: {class_name}"
                    )
                    break  # Только по одному изображению на класс
    
    # Итоги
    end_time = time.time()
    duration = end_time - start_time
    print("\\n" + "="*60)
    print("🎉 ПАЙПЛАЙН ЗАВЕРШЕН!")
    print("="*60)
    print(f"⏱️  Общее время выполнения: {duration:.1f} секунд ({duration/60:.1f} минут)")
    print(f"📅 Завершено: {time.strftime('%H:%M:%S')}")
    
    print("\\n📁 РЕЗУЛЬТАТЫ:")
    results = [
        f"• Модель: runs/classify/train/weights/best.pt",
        f"• Графики обучения: runs/classify/train/results.png", 
        f"• Логи: runs/classify/train/",
        f"• Предсказания: runs/classify/predict/",
        f"• Конфигурация: data.yaml"
    ]
    
    for result in results:
        print(f"  {result}")
    
    print("\\n🚀 ДАЛЬНЕЙШИЕ ДЕЙСТВИЯ:")
    next_steps = [
        "• Для отдельных предсказаний: python scripts/04_predict.py --model best.pt --source your_image.jpg",
        "• Для переобучения: удалите папку runs/classify/train/ и запустите снова",
        "• Для анализа: python scripts/09_analyze_results.py",
        "• Для тестирования: python scripts/08_test_predictions.py --comprehensive"
    ]
    
    for step in next_steps:
        print(f"  {step}")

def main():
    """Основная функция"""
    try:
        run_full_pipeline()
    except KeyboardInterrupt:
        print("\\n❌ Пайплайн прерван пользователем")
    except Exception as e:
        print(f"\\n❌ Критическая ошибка в пайплайне: {e}")
        print("💡 Проверьте структуру проекта и наличие данных")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Подготовка данных NIH ChestX-ray для проекта
"""

import os
import pandas as pd
import sys

# Добавляем пути
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def ensure_nih_metadata():
    """Гарантирует наличие правильных метаданных NIH"""
    metadata_path = "data/nih/Data_Entry_2017.csv"
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Если файл существует, проверяем его валидность
    if os.path.exists(metadata_path):
        print("📁 Файл метаданных существует, проверяем валидность...")
        df = validate_and_read_metadata(metadata_path)
        if df is not None:
            return df
    
    print("📝 Создаем реалистичные демо метаданные NIH...")
    return create_realistic_demo_metadata(metadata_path)

def validate_and_read_metadata(file_path):
    """Проверяет и читает метаданные, возвращает None если файл невалидный"""
    try:
        # Сначала проверим содержимое файла
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(3)]
        
        # Проверяем что это не HTML
        if any('<!DOCTYPE' in line or '<html' in line for line in first_lines):
            print("❌ Файл содержит HTML, а не CSV данные")
            return None
        
        # Пробуем прочитать как CSV
        df = pd.read_csv(file_path, nrows=5)  # Читаем только первые 5 строк для проверки
        
        # Проверяем наличие обязательных колонок
        required_columns = ['Image Index', 'Finding Labels']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Отсутствуют обязательные колонки: {missing_columns}")
            return None
        
        # Если все ок, читаем весь файл
        df = pd.read_csv(file_path)
        print(f"✅ Файл валиден! Строк: {len(df)}")
        return df
        
    except Exception as e:
        print(f"❌ Ошибка при проверке файла: {e}")
        return None

def create_realistic_demo_metadata(file_path):
    """Создает реалистичные демонстрационные метаданные NIH"""
    print("🎯 СОЗДАЕМ РЕАЛИСТИЧНЫЕ ДЕМО ДАННЫЕ NIH CHESTX-RAY...")
    
    # Создаем данные в точном соответствии с форматом NIH ChestX-ray14
    total_images = 200
    
    # Распределение классов как в реальном датасете
    findings_distribution = {
        'No Finding': 140,      # 70% - нормальные снимки
        'Pneumonia': 20,        # 10%
        'Cardiomegaly': 15,     # 7.5%
        'Effusion': 12,         # 6%
        'Nodule': 8,            # 4%
        'Atelectasis': 5        # 2.5%
    }
    
    # Создаем список всех findings
    all_findings = []
    for finding, count in findings_distribution.items():
        all_findings.extend([finding] * count)
    
    demo_data = {
        'Image Index': [f'{i:08d}_000.png' for i in range(1, total_images + 1)],
        'Finding Labels': all_findings,
        'Follow-up #': [i % 6 for i in range(total_images)],
        'Patient ID': [100000 + i for i in range(total_images)],
        'Patient Age': [20 + (i * 37) % 60 for i in range(total_images)],  # Случайный возраст 20-80
        'Patient Gender': ['M' if i % 2 == 0 else 'F' for i in range(total_images)],
        'View Position': ['PA' if i % 3 == 0 else 'AP' for i in range(total_images)],
        'OriginalImage[Width': [1024] * total_images,
        'OriginalImage[Height': [1024] * total_images,
        'OriginalImagePixelSpacing[x': [0.2] * total_images,
        'OriginalImagePixelSpacing[y': [0.2] * total_images,
    }
    
    df = pd.DataFrame(demo_data)
    df.to_csv(file_path, index=False)
    
    print(f"✅ Демо метаданные созданы! Строк: {len(df)}")
    print("📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    for finding, count in findings_distribution.items():
        print(f"   {finding}: {count} изображений ({count/total_images*100:.1f}%)")
    
    return df

def analyze_and_filter_data(df):
    """Анализирует и фильтрует данные"""
    print("\n📊 АНАЛИЗ ДАННЫХ NIH:")
    print(f"Всего записей: {len(df):,}")
    
    # Анализ классов
    print("\n🎯 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    class_distribution = df['Finding Labels'].value_counts()
    for finding, count in class_distribution.items():
        percentage = count / len(df) * 100
        print(f"   {finding}: {count} ({percentage:.1f}%)")
    
    # Фильтруем нормальные снимки
    normal_images = df[df['Finding Labels'] == 'No Finding']
    print(f"\n📈 Нормальных снимков (No Finding): {len(normal_images):,}")
    
    if len(normal_images) == 0:
        print("⚠️  Внимание: нет нормальных снимков!")
        print("   Используем первые 50 снимков как нормальные для демо")
        normal_images = df.head(50)
    
    # Сохраняем список нормальных изображений
    output_file = "data/nih/normal_images_list.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    normal_images['Image Index'].to_csv(output_file, index=False, header=False)
    print(f"✅ Список нормальных изображений сохранен: {output_file}")
    
    # Сохраняем полную информацию о классах
    class_info_file = "data/nih/class_distribution.csv"
    class_distribution.to_csv(class_info_file, header=['Count'])
    print(f"✅ Распределение классов сохранено: {class_info_file}")
    
    return normal_images



    with open('data/nih/README.md', 'w', encoding='utf-8') as f:
        f.write(docs_content)
    print("✅ Документация создана: data/nih/README.md")

def main():
    print("📊 ПОДГОТОВКА ДАННЫХ NIH CHESTX-RAY")
    print("=" * 50)
    
    # Гарантируем наличие правильных метаданных
    df = ensure_nih_metadata()
    
    # Анализируем и фильтруем данные
    normal_images = analyze_and_filter_data(df)
    
  
    

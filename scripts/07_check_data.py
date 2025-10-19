#!/usr/bin/env python3
"""
Проверка реальных данных перед обучением
"""

import os
import glob

def check_real_data():
    print("🔍 ПРОВЕРКА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 40)
    
    folders = {
        'normal': 'data/images/train/normal',
        'clavicle_fracture': 'data/images/train/clavicle_fracture', 
        'foreign_body': 'data/images/train/foreign_body'
    }
    
    total_files = 0
    
    for class_name, folder_path in folders.items():
        if os.path.exists(folder_path):
            # Ищем все изображения
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
            print(f"📁 {class_name}: {len(image_files)} файлов")
            
            total_files += len(image_files)
        else:
            print(f"❌ {class_name}: папка не существует")
    
    print(f"\n📊 ВСЕГО ФАЙЛОВ: {total_files}")
    
    if total_files == 0:
        print("\n💡 ДАННЫХ НЕТ! Загрузите изображения:")
        print("1. Откройте файловый менеджер Colab (иконка папки слева)")
        print("2. Нажмите 'Upload' и выберите файлы")
        print("3. Перетащите файлы в соответствующие папки")
        return False
    elif total_files < 10:
        print("\n⚠️  Мало данных для обучения. Добавьте больше изображений.")
        return False
    else:
        print("\n✅ Данных достаточно для обучения!")
        return True

if __name__ == "__main__":
    check_real_data()

# Детекция людей на видео

Обнаружение людей на видео с использованием модели YOLOv11.

## Описание

Программа обрабатывает видеофайл и выполняет детекцию людей (класс person) на каждом кадре с помощью предобученной модели YOLOv11. Результаты детекции отрисовываются поверх видео в виде ограничивающих рамок (bounding boxes) с указанием названия класса и уверенности (confidence) модели для каждого обнаруженного объекта.

## Выбор модели

Чтобы выбрать оптимальный размер модели, было проведено тестирование на размеченном датасете размером в 81 изображение.

Тестирование проводилось на Nvidia RTX 3070, при порогах Confidence = 0.3 и IoU = 0.5.

Результаты тестирования
| Model     | mAP50  | Precision  | Recall  | Inference (ms)  |
|-----------|--------|------------|---------|-----------------|
| YOLOv11m  | 0.678  | 0.945      | 0.4     | 17.9            |
| YOLOv11l  | 0.69   | 0.967      | 0.408   | 19.8            |
| YOLOv11x  | 0.685  | 0.961      | 0.407   | 24.0            |

Тестирование показало что модель YOLOv11l, показала лучшие результаты по точности детекции и скорости инференса и была выбрана для обработки видео.

## Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/vivch1k/crowd-detection.git
cd crowd-detection
```
2. Создайте и активируйте виртуальное окружение:
```bash
# Windows
python -m venv .venv 
.venv\Scripts\activate 

# Linux / MacOS
python3 -m venv .venv 
source .venv/bin/activate
```
3. Установите зависимости:

В зависимости от вашего оборудования выберите нужный вам способ установки pytorch. Результат работы будет **одинаковый**, отличается только скорость обработки.
```bash
# Windows (CUDA)
pip install typing-extensions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Linux (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Linux (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Windows / MacOS (CPU)
pip install torch torchvision

# Обязательно для всех
pip install -r requirements.txt
```

## Запуск
#### Обработка видео
Отрисовывает обнаруженных людей и сохраняет аннотированное видео.
```bash
# Windows
python -m src.main

# Linux / MacOS
python3 -m src.main
```
Результат сохраняется в ```data/predict_crowd.mp4```
При первом запуске будут скачены веса модели ```yolo11l.pt``` в корень проекта.
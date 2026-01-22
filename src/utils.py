import supervision as sv
import torch


def annotate_frame(box_annt, label_annt, frame, results):
    """
    Выполняет отрисовку результатов детекции на кадре видео.

    Функция преобразует результаты инференса модели YOLO в формат
    библиотеки supervision и отрисовывает bounding box'ы и подписи
    (класс + уверенность) на изображении.

    Args:
        box_annt: Аннотатор для отрисовки bounding box'ов.
        label_annt: Аннотатор для отрисовки текстовых подписей.
        frame (numpy.ndarray): Исходный кадр видео.
        results: Результаты инференса модели YOLO для одного кадра.

    Returns:
        numpy.ndarray: Кадр с отрисованными детекциями.
    """
    class_names = results.names
    labels = [
        f"{class_names[int(cls.item())]} | {conf:.2f}"
        for cls, conf in zip(results.boxes.cls, results.boxes.conf)
    ]

    detections = sv.Detections.from_ultralytics(results)

    annotator = box_annt.annotate(frame, detections)
    annotator = label_annt.annotate(annotator, detections, labels)

    return annotator


def set_seed(seed=42):
    """
    Устанавливает фиксированное значение seed для обеспечения
    воспроизводимости результатов.

    Функция задаёт seed для PyTorch и, при наличии GPU,
    для всех CUDA-устройств.

    Args:
        seed (int, optional): Значение seed. По умолчанию 42.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Определяет доступное устройство для инференса модели.

    Приоритет:
    1. CUDA (GPU NVIDIA)
    2. MPS (Apple Silicon)
    3. CPU

    Returns:
        str: Строковое обозначение устройства ('cuda:0', 'mps' или 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.mps.is_available():
        return "mps"
    else:
        return "cpu"

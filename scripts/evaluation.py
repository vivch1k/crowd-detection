from ultralytics import YOLO
import polars as pl

from src.utils import set_seed


data_path = "data/test_yolov11/data.yaml"
project_name = "data/evaluation"
model_name = "yolo11l.pt"

save_path = f"{project_name}/{model_name.split('.')[0]}"

CONF_THR = 0.3
IOU_THR = 0.8
PERSON_CLASS = 0


def save_results(metrics, save_path):
    """
    Сохраняет метрики качества модели в CSV-файл.

    Извлекает основные метрики из объекта Ultralytics Metrics,
    добавляет информацию о времени инференса и сохраняет результат
    в формате CSV для последующего анализа.

    Args:
        metrics: Объект метрик, возвращаемый методом model.val().
        save_path (str): Путь для сохранения результатов.
    """
    df = metrics.to_df()
    df = df.with_columns([
        pl.lit(metrics.speed["inference"]).alias("inference_ms")
    ])
    df.write_csv(f"{save_path}/results.csv")


if __name__ == "__main__":
    set_seed(42)
    model = YOLO(model_name)

    metrics = model.val(
        data=data_path,
        conf=CONF_THR,
        iou=IOU_THR,
        classes=[PERSON_CLASS],
        device="cuda:0",
        save_dir=save_path,
        save_txt=True,
        save_conf=True,
    )

    save_results(metrics, save_path)

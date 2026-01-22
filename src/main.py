import cv2
from ultralytics import YOLO
import supervision as sv

from src.utils import annotate_frame, set_seed, get_device
from src.config import CONFIG

video_path = "data/crowd.mp4"
output_path = "data/predict_crowd.mp4"

if __name__ == "__main__":
    set_seed(42)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    device = get_device()

    model = YOLO(CONFIG["model"])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.Color.from_hex("#FF00BB"),
        text_scale=0.5,
        text_padding=3,
        text_position=sv.Position.TOP_CENTER,
    )
    round_box_annotator = sv.RoundBoxAnnotator(
        color=sv.Color.from_hex("#FF00BB"), thickness=1
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            source=frame,
            conf=CONFIG["conf"],
            iou=CONFIG["iou"],
            classes=[CONFIG["person_class"]],
            save=False,
            device=device,
        )[0]

        annotator = annotate_frame(
            round_box_annotator,
            label_annotator,
            frame,
            results
        )

        out.write(annotator)

    cap.release()
    out.release()

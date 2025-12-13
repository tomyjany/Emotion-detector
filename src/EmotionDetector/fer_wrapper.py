from EmotionDetector.interface import EmotionDetector
from fer import FER
import cv2
from typing import List, Dict, Any, Optional

from rich.progress import Progress

class FERWrapper(EmotionDetector):
    """
    FER wrapper that assumes a single combined video:
    - full HD frame (e.g. 1920x1080)
    - webcam PiP is in the bottom-left corner, size ~500x280

    We crop that region and run FER only there.
    """

    def __init__(self, roi_width: int = 500, roi_height: int = 280) -> None:
        self.detector = FER(mtcnn=True)  # MTCNN face detector (more accurate)
        self.roi_width = roi_width
        self.roi_height = roi_height

    def detect_emotion(self, path: str) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0  # fallback

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

        # Bottom-left ROI where webcam lives
        roi_w = min(self.roi_width, width)
        roi_h = min(self.roi_height, height)
        roi_x = 0
        roi_y = height - roi_h  # bottom-left corner

        results: List[Dict[str, Any]] = []
        frame_index = 0

        # Rich progress bar over frames
        with Progress() as progress:
            task = progress.add_task(
                "[green]Running FER on webcam region...",
                total=total_frames if total_frames and total_frames > 0 else None,
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Crop to webcam region
                roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

                # Convert BGR → RGB for FER
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                # Run FER only on this ROI
                detections = self.detector.detect_emotions(roi_rgb)

                if detections:
                    # Assume one face in webcam – take the first detection
                    d = detections[0]
                    x, y, w_box, h_box = d["box"]
                    emotions = d["emotions"]  # dict of emotion -> probability

                    # Bounding boxes: in ROI coords + full-frame coords
                    full_x = int(roi_x + x)
                    full_y = int(roi_y + y)
                    full_box = [full_x, full_y, int(w_box), int(h_box)]

                    record: Dict[str, Any] = {
                        "frame_index": frame_index,
                        "timestamp": frame_index / fps,
                        "roi_box": [int(x), int(y), int(w_box), int(h_box)],
                        "box": full_box,
                    }
                    record.update(emotions)
                    results.append(record)

                frame_index += 1
                progress.update(task, advance=1)

        cap.release()
        return results


def get_fer_model(name: str = "fer") -> EmotionDetector:
    if name.lower() == "fer":
        return FERWrapper()
    else:
        raise ValueError(f"Model '{name}' is not supported.")

from EmotionDetectorApp.emotion_detector import EmotionDetector
from fer import Video, FER

class FERWrapper(EmotionDetector):
    def __init__(self):
    self.detector = FER(mtcnn=True)          # MTCNN face detector (more accurate)

    def detect_emotion(self, path: str) -> dict:
        video = Video(path)
        raw = video.analyze(self.detector, display=False)
        df = video.to_pandas(raw)                      # per-frame results → pandas DataFrame
        return df.to_dict(orient='records')


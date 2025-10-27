from abc import ABC, abstractmethod
class EmotionDetector(ABC):
    @abstractmethod
    def detect_emotion(self, path: str) -> dict:
        pass

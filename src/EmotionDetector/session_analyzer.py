from __future__ import annotations

from rich.progress import Progress
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from EmotionDetector.interface import EmotionDetector as EmotionDetectorInterface
from EmotionDetector.fer_wrapper import get_fer_model


def _get_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        fps = 25.0  # sensible fallback
    return float(fps)


@dataclass
class EmotionEvent:
    start_time: float
    end_time: float
    peak_time: float
    dominant_emotion: str
    peak_score: float


class EmotionSessionAnalyzer:
    """
    Session-level analyzer for a single combined video that contains:
      * screen content
      * webcam in bottom-left (FERWrapper crops it automatically)
    """

    def __init__(
        self,
        detector: Optional[EmotionDetectorInterface] = None,
        engagement_window_sec: float = 30.0,
        big_emotion_zscore: float = 1.5,
        big_emotion_min_score: float = 0.6,
    ) -> None:
        self.detector = detector or get_fer_model()
        self.engagement_window_sec = engagement_window_sec
        self.big_emotion_zscore = big_emotion_zscore
        self.big_emotion_min_score = big_emotion_min_score

    def analyze(
        self,
        video_path: str,
        frame_skip: int = 1,
    ) -> pd.DataFrame:
        """
        Run FER on `video_path` (combined video), align frames to timestamps,
        and compute per-frame features.

        Returns a DataFrame with columns including:
          - frame_index
          - timestamp (seconds)
          - per-emotion probabilities (angry/happy/...)
          - dominant_emotion, dominant_score
          - <emotion>_z   (z-score per emotion)
          - engagement_raw, engagement_ma
          - high_engagement (bool)
          - big_emotion (bool)
        """
        raw = self.detector.detect_emotion(video_path)
        df = pd.DataFrame(raw)
        if df.empty:
            raise ValueError("FER returned no frames for this video.")

        # Normalize frame index
        if "frame_index" not in df.columns:
            if "frame" in df.columns:
                df.rename(columns={"frame": "frame_index"}, inplace=True)
            else:
                df["frame_index"] = np.arange(len(df))

        # Optional frame skipping to speed things up
        if frame_skip > 1:
            df = df[df["frame_index"] % frame_skip == 0].reset_index(drop=True)

        # Ensure timestamps
        if "timestamp" not in df.columns:
            fps = _get_fps(video_path)
            df["timestamp"] = df["frame_index"] / fps

        # Emotion columns
        possible = ["angry", "anger", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        emotion_cols = [c for c in df.columns if c in possible]

        # In case emotions were nested (shouldn't be now, but just in case)
        if not emotion_cols and "emotions" in df.columns:
            emo_df = df["emotions"].apply(pd.Series)
            df = pd.concat([df.drop(columns=["emotions"]), emo_df], axis=1)
            emotion_cols = [c for c in emo_df.columns if c in possible]

        if "anger" in emotion_cols and "angry" not in emotion_cols:
            df.rename(columns={"anger": "angry"}, inplace=True)
            emotion_cols = [c if c != "anger" else "angry" for c in emotion_cols]

        if not emotion_cols:
            raise ValueError(
                "Could not find emotion probability columns in FER output "
                f"(looked for {possible})."
            )

        # Dominant emotion per frame
        df["dominant_emotion"] = df[emotion_cols].idxmax(axis=1)
        df["dominant_score"] = df[emotion_cols].max(axis=1)

        # Z-score normalization per emotion (session-wise)
        for col in emotion_cols:
            mean = float(df[col].mean())
            std = float(df[col].std())
            if std == 0.0:
                df[f"{col}_z"] = 0.0
            else:
                df[f"{col}_z"] = (df[col] - mean) / std

        # Engagement: how far from neutral (1 - neutral probability)
        if "neutral" in emotion_cols:
            df["engagement_raw"] = 1.0 - df["neutral"]
        else:
            # fallback: just use the strongest emotion probability
            df["engagement_raw"] = df[emotion_cols].max(axis=1)

        # Rolling mean engagement (for anti-doomscroll window)
        fps = _get_fps(video_path)
        window_frames = max(1, int(self.engagement_window_sec * fps / max(frame_skip, 1)))
        df["engagement_ma"] = df["engagement_raw"].rolling(
            window_frames, min_periods=1, center=True
        ).mean()

        # High engagement = top 10% of engagement_ma
        high_thr = float(df["engagement_ma"].quantile(0.9))
        df["high_engagement"] = df["engagement_ma"] >= high_thr

        # Big emotions: strong outliers in their own z-score + decent raw prob
        df["big_emotion"] = False
        for col in emotion_cols:
            zcol = f"{col}_z"
            if zcol not in df.columns:
                continue
            mask = (
                (df["dominant_emotion"] == col)
                & (df[zcol] >= self.big_emotion_zscore)
                & (df[col] >= self.big_emotion_min_score)
            )
            df.loc[mask, "big_emotion"] = True

        return df

    def get_events(self, df: pd.DataFrame, min_duration_sec: float = 1.0) -> List[Dict[str, Any]]:
        """
        Collapse consecutive 'big_emotion' frames into events.
        """
        if df.empty:
            return []

        # Estimate FPS from timestamps
        dt = df["timestamp"].diff().median()
        if pd.isna(dt) or dt <= 0:
            fps_est = 25.0
        else:
            fps_est = 1.0 / float(dt)

        min_frames = max(1, int(min_duration_sec * fps_est))

        events: List[EmotionEvent] = []

        in_event = False
        start_idx: Optional[int] = None
        # Rich progress bar over rows
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Aggregating big emotion events...",
                total=len(df),
            )
            for idx, row in df.iterrows():
                if row["big_emotion"] and not in_event:
                    in_event = True
                    start_idx = idx
                elif not row["big_emotion"] and in_event:
                    end_idx = idx - 1
                    seg = df.loc[start_idx:end_idx]
                    if len(seg) >= min_frames:
                        peak_row = seg.iloc[seg["dominant_score"].values.argmax()]
                        events.append(
                            EmotionEvent(
                                start_time=float(seg["timestamp"].iloc[0]),
                                end_time=float(seg["timestamp"].iloc[-1]),
                                peak_time=float(peak_row["timestamp"]),
                                dominant_emotion=str(peak_row["dominant_emotion"]),
                                peak_score=float(peak_row["dominant_score"]),
                            )
                        )
                    in_event = False

                progress.update(task, advance=1)

        # Trailing event to end of video
        if in_event and start_idx is not None:
            seg = df.loc[start_idx:]
            if len(seg) >= min_frames:
                peak_row = seg.iloc[seg["dominant_score"].values.argmax()]
                events.append(
                    EmotionEvent(
                        start_time=float(seg["timestamp"].iloc[0]),
                        end_time=float(seg["timestamp"].iloc[-1]),
                        peak_time=float(peak_row["timestamp"]),
                        dominant_emotion=str(peak_row["dominant_emotion"]),
                        peak_score=float(peak_row["dominant_score"]),
                    )
                )

        # Convert dataclasses to plain dicts
        return [e.__dict__ for e in events]

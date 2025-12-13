# src/analyzer_app.py
import argparse
import json
from pathlib import Path

import pandas as pd

from EmotionDetector.session_analyzer import EmotionSessionAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FER analysis on a single combined video (screen + webcam bottom-left)."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to combined video.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis_output",
        help="Where to save frame-level data and events.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Analyze every Nth frame to speed up (default=1 = all frames).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = EmotionSessionAnalyzer()
    df = analyzer.analyze(
        video_path=args.video,
        frame_skip=args.frame_skip,
    )
    events = analyzer.get_events(df)

    session_name = Path(args.video).stem

    df_path = out_dir / f"{session_name}_frames.parquet"
    events_path = out_dir / f"{session_name}_events.json"
    summary_path = out_dir / f"{session_name}_summary.json"
    analysis_json_path = out_dir / f"{session_name}_analysis.json"

    # Save frame-level data as parquet
    df.to_parquet(df_path, index=False)

    # Save events as JSON
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    # Quick summary for the session
    emotion_dist = df["dominant_emotion"].value_counts(normalize=True).to_dict()
    summary = {
        "video": args.video,
        "num_frames": int(len(df)),
        "emotions_distribution": emotion_dist,
        "num_events": len(events),
        "avg_engagement": float(df["engagement_ma"].mean()),
        "high_engagement_fraction": float(df["high_engagement"].mean()),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Unified analysis JSON: frames + events + summary
    analysis_payload = {
        "frames": df.to_dict(orient="records"),
        "events": events,
        "summary": summary,
    }
    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    print(f"[OK] Frame-level data → {df_path}")
    print(f"[OK] Events → {events_path}")
    print(f"[OK] Summary → {summary_path}")
    print(f"[OK] Unified analysis JSON → {analysis_json_path}")


if __name__ == "__main__":
    main()

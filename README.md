# Emotion-Detector

Prototype backend for an anti-doomscrolling application.  
Given a recording of a social media session **with the user’s face visible in a small webcam window**, it:

- runs facial emotion recognition (FER) over the face,
- computes a per-frame “engagement” signal (how emotionally active the face is),
- detects “big emotion” events and high-engagement periods,
- exposes everything via:
  - a CLI analyzer (offline batch analysis),
  - a Streamlit web app (interactive exploration).

---

## 1. Installation

Create and activate a virtual environment (recommended), then install:

```bash
pip install .
```

This will install the `EmotionDetector` package and its dependencies (FER, OpenCV, etc.).

---

## 2. Video format & assumptions

The current prototype makes several assumptions about the input video.  
If these are not respected, face detection / emotions may fail or be noisy.

### 2.1 Single combined video

- Input is **one video file** that contains:
  - the **screen content** (what the user is looking at),
  - a **webcam picture-in-picture (PiP)** with the user’s face.

We do **not** use two separate videos; everything is in this single file.

### 2.2 Webcam position & size

The webcam PiP is assumed to be:

- in the **bottom-left corner** of the frame,
- with a **maximum size of 500×280 pixels**.

This logic lives in `FERWrapper`:

```python
class FERWrapper(EmotionDetector):
    """FER wrapper that assumes a single combined video:
    - full HD frame (e.g. 1920x1080)
    - webcam PiP is in the bottom-left corner, size ~500x280

    We crop that region and run FER only there.
    """

    def __init__(self, roi_width: int = 500, roi_height: int = 280) -> None:
        self.detector = FER(mtcnn=True)  # MTCNN face detector (more accurate)
        self.roi_width = roi_width
        self.roi_height = roi_height
```

Concretely, for each frame:

- we read the full frame,
- we compute a **region of interest (ROI)**:

  - width  = `min(roi_width, frame_width)`  
  - height = `min(roi_height, frame_height)`  
  - x      = `0`  
  - y      = `frame_height - roi_height`  → bottom-left corner,

- we crop that ROI and run FER only inside this small window.

### 2.3 Resolution & duration

- The code is written with **full HD** (1920×1080) in mind, but it will also work with other resolutions as long as:
  - the webcam **really is** in the bottom-left corner,
  - the face fits roughly inside the 500×280 ROI.
- Any video format that OpenCV and MoviePy can read should work (`.mp4`, `.mov`, `.avi`, `.mkv`, …).
- Very long or huge files will work but obviously take longer to analyze.

---

## 3. Running the CLI analyzer

The CLI analyzer runs FER over the video, computes engagement, detects big emotion events, and writes results to files in an output directory.

From the project root:

```bash
python -m EmotionDetector --video path/to/video.mp4
```

> If you prefer calling the script directly, you can also use the analyzer module:
>
> ```bash
> python -m analyzer_app --video path/to/video.mp4
> ```

By default, outputs are written into an `analysis_output/` directory:

- `<video_name>_frames.parquet`  
  – per-frame data (emotion probabilities, engagement, flags, …)
- `<video_name>_events.json`  
  – list of “big emotion” events (start / end / peak time, dominant emotion, score)
- `<video_name>_summary.json`  
  – simple summary (number of frames, emotion distribution, average engagement, etc.)
- `<video_name>_analysis.json`  
  – unified JSON with:
    - `frames`: list of per-frame records,
    - `events`: list of events,
    - `summary`: summary dict.

You can inspect these with any JSON/Parquet viewer or load them in Python for further analysis.

---

## 4. Streamlit web app

The Streamlit app is an interactive frontend to:

- analyze a new video, or  
- load precomputed analysis JSONs and explore them.

### 4.1 Starting the app

From the project root:

```bash
streamlit run src/streamlit_app.py
```

> If your videos are large (hundreds of MB), you may want to increase Streamlit’s upload limit using a config file or command-line flag:
>
> ```bash
> streamlit run src/streamlit_app.py --server.maxUploadSize=1024
> ```
>
> (1024 MB = 1 GB).

Then open the URL Streamlit prints in your terminal (usually http://localhost:8501).

### 4.2 Modes in the app

Once the app is running, you’ll see two modes:

#### Mode 1 – Analyze new video

- Upload a combined video (screen + webcam in bottom-left).
- The app:
  - saves it to a temp file,
  - runs the `EmotionSessionAnalyzer`,
  - caches the result so reloading the page doesn’t recompute everything.

You’ll see:

- the video player,
- an **engagement over time** chart,
- a simple **“doomscrolling monitor”** (average engagement, fraction of time in high engagement),
- a table of **big emotion events** with buttons to play short clips around each peak,
- a **per-frame emotion explorer** with a slider and an emotion bar chart.

#### Mode 2 – Load existing analysis (JSON)

- Upload a `<video_name>_analysis.json` file produced by the CLI.
- Optionally also upload the original video to enable playback and event clips.
- The app skips FER and uses the precomputed data, which is much faster for big files.

---

## 5. Conceptual overview of the analysis

High-level steps the backend performs:

1. **Run FER on each frame** in the webcam ROI  
   → emotion probability distribution over 7 emotions.
2. **Build a per-frame DataFrame**:
   - emotion probabilities,
   - dominant emotion & intensity,
   - engagement signal (`1 - neutral`),
   - smoothed engagement (`engagement_ma`).
3. **Normalize emotions over time** (per emotion, per session) to detect unusually strong frames.
4. **Mark “big emotion” frames** where:
   - an emotion is dominant,
   - its value is high relative to the session baseline.
5. **Group consecutive big-emotion frames** into events (start/end/peak).
6. **Summarize engagement** over the whole session:
   - average engagement,
   - fraction of time spent in high-engagement states.
7. **Expose everything** via:
   - CLI outputs (Parquet/JSON),
   - Streamlit UI for interactive exploration.

This is a prototype of the emotion backend for a future real-time anti-doomscrolling plugin.  
In a full product, the same logic would run continuously in the background and trigger gentle “take a break” suggestions when emotional activity stays high for too long.


import argparse, json, sys
import cv2
from fer import FER

def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("image_path", help="Path to an image with a face")
    # ap.add_argument("--mtcnn", action="store_true", help="Use MTCNN face detector (more accurate)")
    # args = ap.parse_args()
    image_path = "data/smiling_man.jpg"


    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Cannot read image: {image_path}"}))
        sys.exit(1)

    # detector: Haar (default) or MTCNN if --mtcnn
    detector = FER(mtcnn=True)

    results = detector.detect_emotions(img)  # list of {box, emotions}
    if not results:
        print(json.dumps({"faces": []}, indent=2))
        return

    # Optionally pick the largest face as "primary"
    def area(box):  # [x, y, w, h]
        return box[2] * box[3]
    primary = max(results, key=lambda r: area(r["box"]))

    out = {
        "faces": results,                    # all faces with per-emotion probabilities
        "primary_face": primary,             # convenient alias
        "top_emotion": detector.top_emotion(img)  # (label, score)
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

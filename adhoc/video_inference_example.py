from fer import Video, FER
def main():
    # check if file exists
    try:
        with open("data/happy_angry_happy.mp4", "rb"):
            pass
    except FileNotFoundError:
        print("Please download the sample video from into data/happy_angry_happy.mp4")

    video = Video("data/happy_angry_happy.mp4")          # or a webcam stream in the demo script
    detector = FER(mtcnn=True)          # MTCNN face detector (more accurate)

    raw = video.analyze(detector, display=False)   # display=True overlays live preview
    df = video.to_pandas(raw)                      # per-frame results → pandas DataFrame

    df.to_csv("emotions_per_frame.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()


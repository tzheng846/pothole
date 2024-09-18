from utils import read_video,save_video
from ultralytics import YOLO

def main():
    print("Start Main")
    #reading video
    #video_frames = read_video('input_videos\pothole_vid_1.mp4')

    #trained model
    model = YOLO("models\potole_best.pt")
    result = model.track(source = "input_videos\pothole_vid_1.mp4",save = True, conf = 0.7,tracker = "bytetrack.yaml")

    #save video
    #save_video(result,"output_videos/output_video.avi")

if __name__ == "__main__":
    main() 
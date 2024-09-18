from ultralytics import YOLO

model = YOLO('models\potole_best.pt')

results = model.predict('input_videos\pothole_vid_1.mp4',save= True)

print(results[0])

print('================================')
for box in results[0].boxes:
    print(box)
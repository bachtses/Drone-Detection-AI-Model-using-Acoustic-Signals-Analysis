import cv2
import os
import glob

videos_folder = 'videos/no drone'
output_folder = 'screenshots/train/drone'
#output_folder = 'screenshots/train/no drone'

video_extension = '*.avi'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_files = glob.glob(os.path.join(videos_folder, video_extension))

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_name}.")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_interval = int(fps * 0.3)  # screenshot every X sec
    print(f"Processing video: {video_name}")
    frame_count = 0
    saved_frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Finished processing {video_name}. Total frames saved: {saved_frame_count}")
                break
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{video_name}_frame_{saved_frame_count:04d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1
            frame_count += 1
    finally:
        cap.release()

print("Finished processing all videos.")

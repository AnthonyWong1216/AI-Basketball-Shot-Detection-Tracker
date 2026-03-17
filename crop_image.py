import cv2
import os

video_path = "VID_20260314_071156_257.mp4"
output_folder = "my_labels_data"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 每 10 幀抽一張，避免圖片太重複
    if count % 10 == 0:
        cv2.imwrite(f"{output_folder}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
    count += 1

cap.release()
print(f"✅ 抽幀完成！共得到 {saved_count} 張圖片，請開始標註。")

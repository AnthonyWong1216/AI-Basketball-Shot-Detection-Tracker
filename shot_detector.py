# Avi Shah - Basketball Shot Detector/Tracker - July 2023
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.overlay_text = "Waiting..."
        self.model = YOLO("best.pt").to('cuda')
        
        # Uncomment this line to accelerate inference. Note that this may cause errors in some setups.
        #self.model.half()
        
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        # Uncomment line below to use webcam (I streamed to my iPhone using Iriun Webcam)
        # self.cap = cv2.VideoCapture(0)

        # Use video - replace text with your video path
        video_path = "VID_20260307_085309_252.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # 取得不含副檔名的檔名，並加上 _hl.txt
        # 例如：VID_20260307_085309_252_hl.txt
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.txt_filename = f"{base_name}_hl.txt"

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        self.run()

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break

            results = self.model(self.frame, stream=True, device=self.device)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]

                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # Only create ball points if high confidence or near hoop
                    if (conf > .3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)) and current_class == "Basketball":
                        self.ball_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

                    # Create hoop points if high confidence
                    if conf > .5 and current_class == "Basketball Hoop":
                        self.hoop_pos.append((center, self.frame_count, w, h, conf))
                        cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.display_score()
            self.frame_count += 1

            #cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def clean_motion(self):
        # Clean and display ball motion
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
        for i in range(0, len(self.ball_pos)):
            cv2.circle(self.frame, self.ball_pos[i][0], 2, (0, 0, 255), 2)

        # Clean hoop motion and display current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        def format_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # 1. 偵測「出手/上升」 (Up)
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.frame_count
                    
                    # ======= 核心修改：一旦出手(Up)就紀錄 Highlight =======
                    fps = 25  # 請確保與影片 FPS 一致
                    shot_time = self.frame_count / fps
                    
                    # 設定剪輯範圍（前 3 秒，後 2 秒）
                    start_val = max(0, shot_time - 3)
                    end_val = shot_time + 2

                    # 轉換格式
                    start_mm_ss = format_time(start_val)
                    end_mm_ss = format_time(end_val)
                    goal_mm_ss = format_time(shot_time)

                    with open(self.txt_filename, "a", encoding="utf-8") as f:
                        # 輸出格式如：Goal at 01:25 | Cut: 01:22 to 01:27
                        f.write(f"Goal at {goal_mm_ss} | Cut: {start_mm_ss} to {end_mm_ss}\n")

                    print(f">>> 紀錄成功：{goal_mm_ss}")
                    # ======================================================

            # 2. 偵測「下降」 (Down) - 僅用於顯示 UI，不影響紀錄
            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.frame_count

            # 3. 每 10 幀檢查是否完成一次投籃動作並重置
            if self.frame_count % 10 == 0:
                # 如果球已經過 Up 且過了足夠時間（例如 1.5 秒），就強制重置準備下一次
                if self.up and (self.frame_count - self.up_frame > 45):
                    self.up = False
                    self.down = False
                    print("狀態已重置，等待下一次投籃...")

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Add overlay text for shot result if it exists
        if hasattr(self, 'overlay_text'):
            # Calculate text size to position it at the right top corner
            (text_width, text_height), _ = cv2.getTextSize(self.overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)
            text_x = self.frame.shape[1] - text_width - 40  # Right alignment with some margin
            text_y = 100  # Top margin

            # Display overlay text with color (overlay_color)
            cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        self.overlay_color, 6)
            # cv2.putText(self.frame, self.overlay_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()


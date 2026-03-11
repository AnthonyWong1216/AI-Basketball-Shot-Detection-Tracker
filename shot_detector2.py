import os
import cv2
import math
import numpy as np
from ultralytics import YOLO
# 假設你的 utils.py 仍存在
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device

class ShotDetector:
    def __init__(self):
        # 1. 初始化模型與路徑
        video_path = "VID_20260307_085309_252.mp4"
        self.model = YOLO("best.pt").to('cuda')
        self.class_names = ['Basketball', 'Basketball Hoop']
        
        # 2. 獲取影片資訊
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0: self.fps = 30 # 保險設定
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        self.txt_filename = f"{base_name}_hl.txt"
        
        # 3. 初始化變數
        self.ball_pos = []
        self.hoop_pos = []
        self.frame_count = 0
        self.up = False
        self.up_frame = 0
        
        print(f"🚀 開始處理: {video_path} (FPS: {self.fps})")
        self.run_batch()

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def run_batch(self):
        # 使用 predict(stream=True) 是 YOLO 對 5070 最快的批次讀取方式
        results = self.model.predict(
            source="VID_20260307_085309_252.mp4",
            stream=True, 
            device=0,      # 強制 5070
            conf=0.15,     # 稍微調低門檻增加偵測連貫性
            verbose=False  # 關閉 Log 輸出以加速
        )

        for r in results:
            self.frame_count += 1
            boxes = r.boxes
            
            # 處理每一幀的物體
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                current_class = self.class_names[cls]
                center = (int(x1 + w / 2), int(y1 + h / 2))

                # 邏輯判斷 (保持與你原本一致)
                if current_class == "Basketball" and (conf > 0.3 or (in_hoop_region(center, self.hoop_pos) and conf > 0.15)):
                    self.ball_pos.append((center, self.frame_count, w, h, conf))
                
                if current_class == "Basketball Hoop" and conf > 0.5:
                    self.hoop_pos.append((center, self.frame_count, w, h, conf))

            # 清理座標 (這部分涉及 CPU 計算，GPU 正在後台預取下一幀)
            self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)
            if len(self.hoop_pos) > 1:
                self.hoop_pos = clean_hoop_pos(self.hoop_pos)

            # 偵測出手邏輯
            self.detect_highlights()

            if self.frame_count % 300 == 0:
                print(f"已完成: {self.format_time(self.frame_count/self.fps)}")

        print(f"✅ 處理完成！結果已存入: {self.txt_filename}")
        self.cap.release()

    def detect_highlights(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            if not self.up:
                # 調用你的 detect_up
                if detect_up(self.ball_pos, self.hoop_pos):
                    self.up = True
                    self.up_frame = self.frame_count
                    
                    # 紀錄 Highlight
                    shot_time = self.frame_count / self.fps
                    start_val = self.format_time(max(0, shot_time - 3))
                    end_val = self.format_time(shot_time + 2)
                    goal_val = self.format_time(shot_time)

                    with open(self.txt_filename, "a", encoding="utf-8") as f:
                        f.write(f"Shot at {goal_val} | Cut: {start_val} to {end_val}\n")
                    print(f"📌 紀錄出手: {goal_val}")

            # 重置邏輯 (防止重覆觸發)
            if self.up and (self.frame_count - self.up_frame > self.fps * 1.5): # 冷卻 1.5 秒
                self.up = False

if __name__ == "__main__":
    # 在最上方確保環境變數已設定
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    ShotDetector()
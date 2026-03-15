import os
import glob
import cv2
import math
import numpy as np
from multiprocessing import Pool, set_start_method
from ultralytics import YOLO
# 確保 utils.py 在同目錄
try:
    from utils import detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos, get_device
except ImportError:
    print("⚠️ 錯誤: 找不到 utils.py，請確保該檔案在同一個資料夾內。")

def format_time(seconds):
    """ 將秒數轉換為 mm:ss 格式 """
    seconds = max(0, seconds)
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def process_single_video(video_path):
    """ 
    單個影片的處理邏輯 (由 Pool 調用)
    針對 RTX 5070 優化：imgsz=800, half=True, 並行處理
    """
    device = get_device()
    # 載入模型 (不要手動 model.half()，避免 dtype 衝突)
    model = YOLO("best.pt").to(device)
    
    class_names = ['Basketball', 'Basketball Hoop']
    cap = cv2.VideoCapture(video_path)
    fps = 25
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_filename = f"{base_name}_hl.txt"

    ball_pos = []
    hoop_pos = []
    frame_count = 0
    up = False
    up_frame = 0

    print(f"🛡️ [嚴格偵測模式] 啟動: {video_path}")

    # 使用 'w' 模式重新寫入，確保每次執行都是乾淨的
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"--- YouTube Timestamps for {video_path} ---\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 執行推理 (imgsz=800 提升精度, half=True 啟動 5070 FP16 加速)
            results = model(
                frame, 
                stream=True, 
                device=device, 
                imgsz=800, 
                verbose=False,
                half=True  # 這裡開啟半精度最安全
            )

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    current_class = class_names[cls]
                    center = (int(x1 + w / 2), int(y1 + h / 2))

                    # --- 嚴格門檻設定 ---
                    # 籃球：普通區域 > 0.45，籃框附近 > 0.25
                    if current_class == "Basketball":
                        if conf > 0.45 or (in_hoop_region(center, hoop_pos) and conf > 0.25):
                            ball_pos.append((center, frame_count, w, h, conf))
                    
                    # 籃框：信心值 > 0.65 才認可
                    if current_class == "Basketball Hoop" and conf > 0.65:
                        hoop_pos.append((center, frame_count, w, h, conf))

            # 軌跡清理與籃框定位 (使用 utils 邏輯)
            ball_pos = clean_ball_pos(ball_pos, frame_count)
            if len(hoop_pos) > 1:
                hoop_pos = clean_hoop_pos(hoop_pos)

            # 出手偵測 (嚴格判定：籃球軌跡必須累積超過 5 幀)
            if len(hoop_pos) > 0 and len(ball_pos) > 5:
                if not up:
                    up = detect_up(ball_pos, hoop_pos)
                    if up:
                        up_frame = frame_count
                        shot_time = frame_count / fps
                        
                        # YouTube 格式：偵測時間 - 2秒
                        youtube_ts = format_time(shot_time)
                        f.write(f"{youtube_ts} - Shot\n")
                        f.flush() # 確保即時寫入
                        print(f"📌 [{base_name}] 偵測到出手: {youtube_ts}")

                # 冷卻時間設定為 3 秒 (防止重覆觸發同一球)
                if up and (frame_count - up_frame > fps * 3):
                    up = False

            frame_count += 1
            if frame_count % 1500 == 0:
                print(f"🕒 [{base_name}] 正在分析中... 當前位置: {format_time(frame_count/fps)}")

    cap.release()
    print(f"✅ [{base_name}] 分析完畢。")

if __name__ == "__main__":
    # Windows 必須設定 spawn
    set_start_method('spawn', force=True)
    
    # 搜尋同目錄下所有 mp4
    video_list = glob.glob("*.mp4")
    
    if not video_list:
        print("❌ 找不到影片檔案！")
    else:
        # 4 進程平行處理，RTX 5070 效能全開
        max_parallel = 4
        print(f"🔎 找到 {len(video_list)} 段影片，以 {max_parallel} 併發模式處理...")
        
        with Pool(processes=max_parallel) as pool:
            pool.map(process_single_video, video_list)
        
        print("\n🎉 全部處理完成！請查看對應的 _hl.txt 檔案。")
import os
import glob
import cv2
import torch
from multiprocessing import Pool, set_start_method
from ultralytics import YOLO

try:
    from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos
except ImportError:
    print("⚠️ 警告: 找不到 utils.py")

def format_time(seconds):
    seconds = max(0, seconds)
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def process_single_video(video_path):
    # 1. 載入模型 (RTX 5070 建議用 YOLOv12s 或 v12m)
    model = YOLO("best.pt").to('cuda')
    class_names = ['Basketball', 'Basketball Hoop']
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_filename = f"{base_name}_youtube_timestamps.txt"
    
    ball_pos, hoop_pos = [], []
    frame_count = 0
    up = False
    up_frame = 0

    print(f"🚀 [強化偵測模式] 處理: {video_path}")

    # --- 改動 1：放寬 imgsz 到 960，這是一個 5070 的平衡點 ---
    # --- 改動 2：將 conf 降到極低的 0.05，我們先看到「框」再說 ---
    results = model.track(
        source=video_path,
        stream=True,
        device=0,
        conf=0.05,      # 大放水：只要有 5% 像球就抓出來
        iou=0.5, 
        half=True,
        imgsz=960,      # 1024 如果太慢，960 也是很細緻的
        persist=True,
        augment=False,  # 先關掉增強，看看單純放寬門檻有沒有效（省時間）
        verbose=False,
        save=True       # 重要：這會存下一個帶框的影片在 runs/detect/track/
    )

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"--- YouTube Timestamps for {video_path} ---\n")
        
        for r in results:
            frame_count += 1
            if not r.boxes:
                continue
                
            boxes = r.boxes
            for box in boxes:
                # 取得數據
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                w, h = x2 - x1, y2 - y1
                center = (int(x1 + w / 2), int(y1 + h / 2))

                # 籃球：對籃框附近的球放寬門檻，對一般的球保持 0.25
                # --- 改動 3：邏輯放寬 ---
                if class_names[cls] == "Basketball":
                    # 只要信心 > 0.1 或是靠近籃框就記錄
                    if conf > 0.1 or (in_hoop_region(center, hoop_pos) and conf > 0.05):
                        ball_pos.append((center, frame_count, x2-x1, y2-y1, conf))
                
                if class_names[cls] == "Basketball Hoop" and conf > 0.3:
                    hoop_pos.append((center, frame_count, x2-x1, y2-y1, conf))

            # --- 改動 4：暫時把清理函數註解掉，看看是不是它刪錯了 ---
            # ball_pos = clean_ball_pos(ball_pos, frame_count) 
            if len(hoop_pos) > 1:
                hoop_pos = clean_hoop_pos(hoop_pos)

            # 出手偵測邏輯
            if len(hoop_pos) > 0 and len(ball_pos) > 0:
                if not up and detect_up(ball_pos, hoop_pos):
                    up = True
                    up_frame = frame_count
                    detect_sec = frame_count / fps
                    
                    # 修正輸出：偵測時間 - 2 秒（YouTube 格式）
                    youtube_ts = format_time(detect_sec - 2)
                    f.write(f"{youtube_ts} - Shot Highlight\n")
                    f.flush()
                    print(f"📌 [{base_name}] 成功偵測出手: {youtube_ts}")

                # 冷卻時間 1.5 秒
                if up and (frame_count - up_frame > fps * 1.5):
                    up = False

            if frame_count % 1000 == 0:
                print(f"🕒 [{base_name}] 進度: {format_time(frame_count/fps)}")

    print(f"✅ [{base_name}] 處理完成！")

if __name__ == "__main__":
    # Windows 必要設定
    set_start_method('spawn', force=True)
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    
    videos = glob.glob("*.mp4")
    if not videos:
        print("❌ 找不到影片")
    else:
        # 5070 開 3 個 1024 偵測負載會很高，建議先觀察 GPU 記憶體
        print(f"🔍 找到 {len(videos)} 段影片，以 3 併發模式處理...")
        with Pool(processes=3) as pool:
            pool.map(process_single_video, videos)
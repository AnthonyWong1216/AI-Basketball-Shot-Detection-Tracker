from ultralytics import YOLO
import os

if __name__ == "__main__":
    # 確保 ultralytics 是最新版本以支援 v12
    # 指令: pip install -U ultralytics

    # --- 關鍵修改點 ---
    # 使用 YOLOv12 的預訓練權重。
    # 'yolov12n.pt' 是 Nano 版 (最快)
    # 'yolov12s.pt' 是 Small 版 (更準，適合 5070)
    PRE_TRAINED_MODEL = 'yolov12n.pt' 


    # 1. 載入 YOLOv12 模型
    # 如果本地沒有檔，它會自動從 Ultralytics 伺服器下載
    model = YOLO(PRE_TRAINED_MODEL)

    # 2. 開始訓練
    # data='config.yaml' 保持不變，因為資料集格式與 v8 通用
    results = model.train(
        data='config.yaml', 
        epochs=100, 
        imgsz=640, 
        device=0,      # 強制指定你的 RTX 5070
        batch=16,      # 5070 顯存 12GB，16-32 應該很輕鬆
        plots=True,    # 產生訓練圖表
        save=True,     # 儲存 best.pt
        half=True,     # 5070 支援 FP16 混合精度訓練，速度更快
        workers=8      # 根據你的 CPU 核心數調整，加速資料讀取
    )

    print("✅ YOLOv12 訓練完成！")
    print("🚀 最強模型路徑: runs/detect/train/weights/best.pt")

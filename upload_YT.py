import os
import glob
import pickle
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# 權限範圍：允許上傳影片至 YouTube
SCOPES = ["https://www.googleapis.com"]

def get_authenticated_service():
    creds = None
    # token.pickle 儲存你的登入授權，第一次執行後會自動生成
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
            
    # 如果沒有有效憑證，則啟動登入流程
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # 這裡需要你從 Google Cloud 下載的 json 檔案
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # 儲存憑證以便下次自動登入
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build("youtube", "v3", credentials=creds)

def upload_video(youtube, video_path, description):
    file_name = os.path.basename(video_path)
    print(f"🎬 準備上傳影片: {file_name}")
    
    body = {
        "snippet": {
            "title": file_name,
            "description": description,
            "tags": ["Basketball", "AI Detection", "Highlights"],
            "categoryId": "17" # 體育類別
        },
        "status": {
            "privacyStatus": "private", # 建議先設為私有(private)，確認後再手動公開
            "selfDeclaredMadeForKids": False
        }
    }

    # 設定媒體檔案上傳
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    
    print(f"📤 正在上傳至 YouTube Studio...")
    response = request.execute()
    print(f"✅ 上傳成功！影片 ID: {response.get('id')}")
    print(f"🔗 連結: https://youtu.be{response.get('id')}\n")

if __name__ == "__main__":
    # 1. 取得授權服務
    if not os.path.exists('client_secrets.json'):
        print("❌ 錯誤: 找不到 client_secrets.json！請從 Google Cloud 下載並放在同資料夾。")
    else:
        youtube_service = get_authenticated_service()
        
        # 2. 搜尋同層目錄下的所有 mp4 影片
        video_files = glob.glob("*.mp4")
        
        if not video_files:
            print("❓ 找不到任何 .mp4 檔案。")
        else:
            print(f"🔎 找到 {len(video_files)} 個影片，準備開始上傳...")
            
            for v_path in video_files:
                # 自動搜尋對應的 _youtube_timestamps.txt 檔案
                # 假設檔名格式為: VID_..._youtube_timestamps.txt
                base_name = os.path.splitext(v_path)[0]
                txt_path = f"{base_name}_youtube_timestamps.txt"
                
                # 讀取 Description
                desc_content = "AI Basketball Shot Detection Highlights\n\n"
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        desc_content += f.read()
                    print(f"📝 已讀取時間軸說明檔: {txt_path}")
                else:
                    print(f"⚠️ 找不到對應的說明檔 {txt_path}，將使用預設說明。")
                
                # 執行上傳
                try:
                    upload_video(youtube_service, v_path, desc_content)
                except Exception as e:
                    print(f"❌ 上傳 {v_path} 時發生錯誤: {e}")

            print("🎉 所有任務已完成！")
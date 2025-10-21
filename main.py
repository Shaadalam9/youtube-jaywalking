# main.py — Automated Video Analyzer using mapping.csv
# Description:
# Reads mapping.csv, downloads each video (FTP → YouTube fallback),
# optionally trims based on timestamps, and runs YOLOv11 detection.

import os
import pandas as pd
from dotenv import load_dotenv
from helper_script import Youtube_Helper

# --- Load FTP credentials from .env ---
load_dotenv()
ftp_server = os.getenv("FTP_SERVER")
ftp_username = os.getenv("FTP_USERNAME")
ftp_password = os.getenv("FTP_PASSWORD")

# --- Folder Setup ---
videos_folder = "./videos"
output_folder = "./outputs"
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# --- Load mapping file ---
mapping_file = "./mapping.csv"
if not os.path.exists(mapping_file):
    raise FileNotFoundError("mapping.csv not found! Please place it in the project folder.")

mapping_df = pd.read_csv(mapping_file)

# --- Validate 'videos' column ---
if "videos" not in mapping_df.columns:
    raise ValueError("mapping.csv must contain a 'videos' column.")

print("\n Automated Video Analyzer — Using Mapping File")
print("--------------------------------------------------")

# --- Initialize helper ---
helper = Youtube_Helper()

# --- Main Loop: Process each row in mapping.csv ---
for idx, row in mapping_df.iterrows():
    # Parse multiple video IDs from a single cell
    raw_videos = str(row["videos"]).strip()
    raw_videos = raw_videos.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    video_ids = [v.strip() for v in raw_videos.split(",") if v.strip()]

    city = row.get("city", "Unknown")
    start_time = row.get("start_time", None)
    end_time = row.get("end_time", None)

    try:
        start_time = float(start_time) if pd.notna(start_time) else None
        end_time = float(end_time) if pd.notna(end_time) else None
    except Exception:
        start_time, end_time = None, None

    #  Process each video ID individually
    for video_id in video_ids:
        print(f"\n🎬 Processing video: {video_id} ({city})")
        print(f"Start: {start_time or 'N/A'}s | End: {end_time or 'N/A'}s")

        try:
            #  Try FTP download
            print("📥 Downloading video from FTP...")
            result = helper.download_videos_from_ftp(
                filename=video_id,
                base_url=ftp_server,
                out_dir=videos_folder,
                username=ftp_username,
                password=ftp_password
            )

            #  Fallback — try YouTube if FTP fails
            if not result:
                print(" FTP failed — trying YouTube...")
                result = helper.download_video_with_resolution(vid=video_id, output_path=videos_folder)

            if not result:
                print(f"Skipping {video_id} — could not download from FTP or YouTube.")
                continue

            #  Successful download → unpack result
            video_path, video_title, resolution, fps = result
            print(f"Downloaded: {video_path} ({resolution}, {fps} FPS)")

            # Step : Trim video (if timestamps available)
            trimmed_path = video_path
            if start_time and end_time and end_time > start_time:
                trimmed_path = os.path.join(videos_folder, f"{video_id}_trimmed.mp4")
                helper.trim_video(video_path, trimmed_path, start_time, end_time)
                print(f"Trimmed segment saved at {trimmed_path}")

            # Step : YOLOv11 Object Detection
            print("Running object detection/tracking...")
            fps_val = helper.get_video_fps(trimmed_path) or 30
            helper.tracking_mode(
                input_video_path=trimmed_path,
                output_video_path=output_folder,
                video_title=os.path.basename(trimmed_path),
                video_fps=int(fps_val),
                seg_mode=False,      # segmentation off
                bbox_mode=True,      # bounding-box detection
                flag=1               # save annotated video
            )

            print(f"Detection complete for {video_id}")
            print(f" Output saved in: {output_folder}\n")

        except Exception as e:
            print(f" Error processing {video_id}: {e}")

print("\n All videos processed successfully using mapping.csv!")

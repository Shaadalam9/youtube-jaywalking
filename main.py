# main.py — Fully Automated FTP + YOLO + Analytics Pipeline

import os
import pandas as pd
from dotenv import load_dotenv
from helper_script import Youtube_Helper
from algorithms import Algorithms
import common

# --- Load environment variables ---
load_dotenv()
ftp_server = os.getenv("FTP_SERVER")
ftp_username = os.getenv("FTP_USERNAME")
ftp_password = os.getenv("FTP_PASSWORD")

# --- Directories ---
input_folder = "./videos"
output_folder = "./outputs"
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# --- Initialize helper and analytics classes ---
helper = Youtube_Helper()
algo = Algorithms()

# --- Load mapping file ---
mapping_path = common.get_configs("mapping")
df_mapping = pd.read_csv(mapping_path)

# --- Flatten all video IDs from mapping.csv ---
all_video_ids = []
for vids in df_mapping["videos"]:
    if isinstance(vids, list):
        all_video_ids.extend(vids)
    elif isinstance(vids, str):
        vids = vids.strip("[] ")
        all_video_ids.extend([v.strip().strip("'\"") for v in vids.split(",") if v.strip()])

all_video_ids = list(set(all_video_ids))  # remove duplicates
print(f"\nTotal videos found in mapping.csv: {len(all_video_ids)}")

print("\n Starting Automated Processing...\n")

for idx, video_id in enumerate(all_video_ids, start=1):
    try:
        print(f"\n[{idx}/{len(all_video_ids)}] 📥 Processing video: {video_id} ...")

        # Step 1: Try FTP first
        result = helper.download_videos_from_ftp(
            filename=video_id,
            base_url=ftp_server,
            out_dir=input_folder,
            username=ftp_username,
            password=ftp_password
        )

        # Step 2: Fallback to YouTube if FTP fails
        if not result:
            print(" FTP not found — trying YouTube...")
            result = helper.download_video_with_resolution(vid=video_id, output_path=input_folder)

        if not result:
            print(f" Skipping {video_id}: could not download.")
            continue

        video_path, video_title, resolution, fps = result
        print(f" Downloaded successfully: {video_path}")

        # Step 3: YOLO Detection
        print(" Running YOLO detection...")
        fps_val = helper.get_video_fps(video_path) or 30
        helper.tracking_mode(
            input_video_path=video_path,
            output_video_path=output_folder,
            video_title=os.path.basename(video_path),
            video_fps=int(fps_val),
            seg_mode=False,
            bbox_mode=True,
            flag=1
        )

        print(f"Detection completed for {video_id}")

        # Step 4: Run Analytics
        tracking_csv = os.path.join(output_folder, f"{video_id}_tracking.csv")
        analytics_csv = os.path.join(output_folder, f"{video_id}_analytics.csv")

        if os.path.exists(tracking_csv):
            df = pd.read_csv(tracking_csv)

            print("Running analytics...")
            crossed = algo.pedestrian_crossing(df, video_id, df_mapping)
            speed_df = algo.calculate_speed_of_crossing(df, fps_val)
            time_df = algo.time_to_cross(df, fps_val)

            # Merge results
            analytics_df = pd.merge(time_df, speed_df, on="person_id", how="outer")
            analytics_df["video_id"] = video_id
            analytics_df["num_crossed"] = len(crossed)
            analytics_df.to_csv(analytics_csv, index=False)

            print(f" Analytics complete for {video_id}")
            print(f" Pedestrians crossed: {len(crossed)}")
            print(f" Saved analytics: {analytics_csv}\n")
        else:
            print(" No tracking data found — skipping analytics.")

    except Exception as e:
        print(f" Error processing {video_id}: {e}")
        continue

print("\n All videos processed successfully!")
print(f"Check results inside: {output_folder}")

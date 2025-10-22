# main.py — Automated Pedestrian Behaviour Detection using YOLOv11
# Uses country + city filter from mapping.csv
#  No manual inputs or prints — fully automated execution
#  Works with existing helper_script, algorithms, and common modules

import os
import pandas as pd
from dotenv import load_dotenv
from helper_script import Youtube_Helper
from algorithms import Algorithms
import common
import logging

# --- Setup logging (no print, all logs go to file) ---
logging.basicConfig(
    filename="run_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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

# --- Filter based on COUNTRY and CITY ---
# ⚙️ Set your desired country and city here:
target_country = "Netherlands"   # change this
target_city = "Eindhoven"        # change this

subset_df = df_mapping[
    (df_mapping["country"].str.lower() == target_country.lower()) &
    (df_mapping["city"].str.lower() == target_city.lower())
]

if subset_df.empty:
    logging.warning(f"No entries found for {target_country}, {target_city}. Exiting pipeline.")
    raise SystemExit()

# --- Flatten all video IDs for selected region ---
all_video_ids = []
for vids in subset_df["videos"]:
    if isinstance(vids, list):
        all_video_ids.extend(vids)
    elif isinstance(vids, str):
        vids = vids.strip("[] ")
        all_video_ids.extend([v.strip().strip("'\"") for v in vids.split(",") if v.strip()])

all_video_ids = list(set(all_video_ids))  # remove duplicates
logging.info(f"Processing {len(all_video_ids)} videos for {target_city}, {target_country}")

# --- Process each video automatically ---
for idx, video_id in enumerate(all_video_ids, start=1):
    try:
        logging.info(f"[{idx}/{len(all_video_ids)}] Starting video ID: {video_id}")

        # Step 1: Try FTP download
        result = helper.download_videos_from_ftp(
            filename=video_id,
            base_url=ftp_server,
            out_dir=input_folder,
            username=ftp_username,
            password=ftp_password
        )

        # Step 2: Fallback to YouTube
        if not result:
            logging.warning(f"{video_id}: FTP not found — switching to YouTube.")
            result = helper.download_video_with_resolution(vid=video_id, output_path=input_folder)

        if not result:
            logging.error(f"{video_id}: Failed to download from FTP or YouTube.")
            continue

        video_path, video_title, resolution, fps = result
        logging.info(f"{video_id}: Download successful ({resolution}, {fps} FPS).")

        # Step 3: YOLOv11 Detection
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
        logging.info(f"{video_id}: YOLO detection completed successfully.")

        # Step 4: Run analytics (behaviour analysis)
        tracking_csv = os.path.join(output_folder, f"{video_id}_tracking.csv")
        analytics_csv = os.path.join(output_folder, f"{video_id}_analytics.csv")

        if os.path.exists(tracking_csv):
            df = pd.read_csv(tracking_csv)
            crossed = algo.pedestrian_crossing(df, video_id, df_mapping)
            speed_df = algo.calculate_speed_of_crossing(df, fps_val)
            time_df = algo.time_to_cross(df, fps_val)

            analytics_df = pd.merge(time_df, speed_df, on="person_id", how="outer")
            analytics_df["video_id"] = video_id
            analytics_df["num_crossed"] = len(crossed)
            analytics_df["country"] = target_country
            analytics_df["city"] = target_city
            analytics_df.to_csv(analytics_csv, index=False)

            logging.info(f"{video_id}: Analytics completed. Saved to {analytics_csv}")
        else:
            logging.warning(f"{video_id}: Tracking file not found, skipping analytics.")

    except Exception as e:
        logging.error(f"{video_id}: Error during processing — {e}")
        continue

logging.info(f" Processing complete for {target_city}, {target_country}. All outputs in {output_folder}")

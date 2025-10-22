# helper_script.py — simplified, production-safe imports
import os
import re
import time
import shutil
import datetime
import pathlib
import json
import yaml
import cv2
import torch
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.video.io.VideoFileClip import VideoFileClip
from ultralytics import YOLO
from custom_logger import CustomLogger
import common
import logging
import subprocess
import sys
import yt_dlp
from collections import defaultdict
from typing import Optional, Set, List, Any

# -------------------------------- Logger & Constants --------------------------------
logger = CustomLogger(__name__)  # use custom logger
logging.getLogger("ultralytics").setLevel(logging.ERROR)  # Show only errors

LINE_THICKNESS = 1
RENDER = False
SHOW_LABELS = False
SHOW_CONF = False
UPGRADE_LOG_FILE = "upgrade_log.json"


# ------------------------------------------------------------------------------------
#                                    Youtube_Helper
# ------------------------------------------------------------------------------------
class Youtube_Helper:
    """Helper class for managing video download, detection, and YOLOv11 tracking."""

    def __init__(self, video_title=None):
        self.tracking_model = common.get_configs("tracking_model")
        self.segment_model = common.get_configs("segment_model")
        self.bbox_tracker = common.get_configs("bbox_tracker")
        self.seg_tracker = common.get_configs("seg_tracker")
        self.resolution = None
        self.mapping = pd.read_csv(common.get_configs("mapping"))
        self.confidence = 0.0
        self.display_frame_tracking = common.get_configs("display_frame_tracking")
        self.display_frame_segmentation = common.get_configs("display_frame_segmentation")
        self.output_path = common.get_configs("videos")
        self.save_annoted_img = common.get_configs("save_annoted_img")
        self.save_tracked_img = common.get_configs("save_tracked_img")
        self.delete_labels = common.get_configs("delete_labels")
        self.delete_frames = common.get_configs("delete_frames")
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")
        self.video_title = video_title or "video"

    # ---------------- Utility & Upgrade Management ----------------
    def set_video_title(self, title):
        self.video_title = title

    def load_upgrade_log(self):
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}
        try:
            with open(UPGRADE_LOG_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    def save_upgrade_log(self, log_data):
        with open(UPGRADE_LOG_FILE, "w") as file:
            json.dump(log_data, file)

    def was_upgraded_today(self, package_name):
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()
        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name):
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name):
        if self.was_upgraded_today(package_name):
            return
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            self.mark_as_upgraded(package_name)
        except subprocess.CalledProcessError:
            self.mark_as_upgraded(package_name)

    # ---------------- FTP Download ----------------
    def download_videos_from_ftp(self, filename, base_url=None, out_dir=".", username=None, password=None,
                                 token=None, timeout=20):
        if not base_url or not username or not password:
            return None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        if not base_url.endswith("/"):
            base_url += "/"

        aliases = ["tue1", "tue2", "tue3"]
        visited = set()

        try:
            with requests.Session() as session:
                session.auth = (username, password)
                session.headers.update({"User-Agent": "ftp-video-downloader/1.0"})

                def fetch(url):
                    try:
                        r = session.get(url, timeout=timeout)
                        r.raise_for_status()
                        return r
                    except requests.RequestException:
                        return None

                def crawl(start_url):
                    stack = [start_url]
                    while stack:
                        url = stack.pop()
                        if url in visited:
                            continue
                        visited.add(url)
                        resp = fetch(url)
                        if resp is None:
                            continue
                        soup = BeautifulSoup(resp.text, "html.parser")
                        for a in soup.find_all("a"):
                            href = a.get("href", "")
                            full = urljoin(base_url, href)
                            if "/files/" in href and filename_with_ext in href:
                                return full
                            if "/browse" in href and href.endswith("/"):
                                stack.append(full)
                    return None

                for alias in aliases:
                    start = urljoin(base_url, f"v/{alias}/browse")
                    found_url = crawl(start)
                    if not found_url:
                        continue
                    os.makedirs(out_dir, exist_ok=True)
                    local_path = os.path.join(out_dir, filename_with_ext)
                    with session.get(found_url, stream=True, timeout=timeout) as r:
                        r.raise_for_status()
                        total = int(r.headers.get("content-length", 0))
                        with open(local_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                               desc=f"Downloading via FTP: {filename_with_ext}") as bar:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))
                    fps = self.get_video_fps(local_path)
                    resolution = self.get_video_resolution_label(local_path)
                    return local_path, filename, resolution, fps
            return None
        except Exception:
            return None

    # ---------------- YouTube Fallback ----------------
    def download_video_with_resolution(self, vid, resolutions=["720p", "480p", "360p", "144p"], output_path="."):
        try:
            if self.update_package and datetime.datetime.today().weekday() == 0:
                self.upgrade_package_if_needed("pytubefix")

            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            youtube_object = YouTube(youtube_url, self.client, on_progress_callback=on_progress)
            for res in resolutions:
                stream = youtube_object.streams.filter(res=res).first()
                if stream:
                    output_file = os.path.join(output_path, f"{vid}.mp4")
                    stream.download(output_path, filename=f"{vid}.mp4")
                    fps = self.get_video_fps(output_file)
                    return output_file, vid, res, fps
            return None
        except Exception:
            return None

    # ---------------- Video Utilities ----------------
    def get_video_fps(self, video_file_path):
        try:
            video = cv2.VideoCapture(video_file_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            video.release()
            return round(fps, 0)
        except Exception as e:
            logger.error(f"Failed to retrieve FPS: {e}")
            return None

    @staticmethod
    def get_video_resolution_label(video_path):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        labels = {144: "144p", 240: "240p", 360: "360p", 480: "480p", 720: "720p",
                  1080: "1080p", 1440: "1440p", 2160: "2160p"}
        return labels.get(height, f"{height}p")

    def trim_video(self, input_path, output_path, start_time, end_time):
        video_clip = VideoFileClip(input_path).subclip(start_time, end_time)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video_clip.close()

    # ---------------- YOLOv11 Tracking ----------------
    def update_track_buffer_in_yaml(self, yaml_path, video_fps):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        config['track_buffer'] = common.get_configs("track_buffer_sec") * video_fps
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def tracking_mode(self, input_video_path, output_video_path, video_title, video_fps,
                      seg_mode=False, bbox_mode=True, flag=0):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if bbox_mode:
            model = YOLO(self.tracking_model)
        elif seg_mode:
            model = YOLO(self.segment_model)
        else:
            return

        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

        os.makedirs(output_video_path, exist_ok=True)
        out_path = os.path.join(output_video_path, f"{video_title}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, video_fps, (frame_width, frame_height))

        progress = tqdm(total=total_frames, desc="Detecting", unit="frames")

        while True:
            success, frame = cap.read()
            if not success:
                break
            try:
                results = model.track(frame, tracker=self.bbox_tracker if bbox_mode else self.seg_tracker,
                                      persist=True, conf=self.confidence, verbose=False, device=device)
                annotated = results[0].plot()
                out.write(annotated)
                if self.display_frame_tracking:
                    cv2.imshow("YOLOv11 Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except Exception as e:
                logger.error(f"Frame error: {e}")
            progress.update(1)

        cap.release()
        out.release()
        progress.close()
        cv2.destroyAllWindows()
        logger.info(f"✅ YOLO detection finished. Output saved to {out_path}")

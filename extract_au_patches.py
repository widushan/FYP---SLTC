import os
import cv2
import pandas as pd
import numpy as np
import glob
import subprocess
from tqdm import tqdm

# Configuration
DATASET_DIR = r"..\..\New PD Videos\Training"
OPENFACE_DIR = r"..\OpenFace_2.2.0_win_x64"
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
OUTPUT_BASE = "Feature Analysis"
PATCH_SIZE = 64
GLOBAL_SIZE = 128

# 18 AUs from Track A / HiVA
FEATURES = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU45",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU20", "AU23",
    "AU24", "AU25", "AU26", "AU17"
]

# Landmark-based ROI Mapping (OpenFace 68 landmarks)
# Format: {AU: [landmark_indices]}
AU_LANDMARK_MAP = {
    "AU01": [21, 22],             # Inner Brow
    "AU02": [17, 26],             # Outer Brow
    "AU04": [21, 22],             # Brow Lowerer (overlap with AU01)
    "AU05": [37, 38, 43, 44],      # Upper Lid
    "AU06": [40, 41, 46, 47],      # Cheek / Lower lid
    "AU07": [37, 38, 40, 41, 43, 44, 46, 47], # Lid Tighten
    "AU09": [31, 35],             # Nose Bridge
    "AU10": [50, 52],             # Upper Lip
    "AU12": [48, 54],             # Lip Corners
    "AU14": [48, 54],             # Dimpler
    "AU15": [48, 54],             # Lip Corners Down
    "AU17": [8, 57, 58],          # Chin
    "AU20": [48, 54],             # Lip Stretch
    "AU23": [48, 54, 51, 57],      # Lip Tighten
    "AU24": [48, 54, 51, 57],      # Lip Press
    "AU25": [60, 64, 62, 66],      # Lips Part
    "AU26": [8, 57],              # Jaw Drop
    "AU45": [36, 39, 42, 45]       # Blink (Eye Corners)
}

def extract_patches():
    print("Starting AU Patch Extraction...")
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    for f in FEATURES:
        for label in ["healthy", "parkinson"]:
            os.makedirs(os.path.join(OUTPUT_BASE, f, label), exist_ok=True)
    for label in ["healthy", "parkinson"]:
        os.makedirs(os.path.join(OUTPUT_BASE, "GLOBAL", label), exist_ok=True)

    classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    for category in classes:
        label = category.lower()
        video_files = glob.glob(os.path.join(DATASET_DIR, category, "*.mp4"))
        
        for video_path in tqdm(video_files, desc=f"Processing {category}"):
            video_name = os.path.basename(video_path)
            temp_csv_dir = "temp_of_patches"
            os.makedirs(temp_csv_dir, exist_ok=True)
            
            # 1. Run OpenFace for landmarks
            cmd = [
                os.path.abspath(OPENFACE_EXE),
                "-f", os.path.abspath(video_path),
                "-out_dir", os.path.abspath(temp_csv_dir),
                "-mloc", "model/main_clnf_general.txt",
                "-landmark2D"
            ]
            
            csv_path = os.path.join(temp_csv_dir, video_name.replace(".mp4", ".csv"))
            if not os.path.exists(csv_path):
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=os.path.abspath(OPENFACE_DIR))
            
            if not os.path.exists(csv_path):
                print(f"Failed to generate CSV for {video_name}")
                continue
                
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            
            # 2. Pick a good frame (e.g., middle of video where success=1)
            valid_frames = df[df['success'] == 1]
            if valid_frames.empty:
                print(f"No successful frames for {video_name}")
                continue
            
            target_row = valid_frames.iloc[len(valid_frames)//2]
            frame_idx = int(target_row['frame'])
            
            # 3. Read the frame from video
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Failed to read frame {frame_idx} from {video_name}")
                continue

            # 4. Crop patches
            h, w, _ = frame.shape
            
            # Extract Global Face (box around all landmarks)
            x_cols = [f"x_{i}" for i in range(68)]
            y_cols = [f"y_{i}" for i in range(68)]
            xs = target_row[x_cols].values
            ys = target_row[y_cols].values
            
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            
            # Expand slightly for global face
            pad = 20
            gx1, gy1 = max(0, min_x-pad), max(0, min_y-pad)
            gx2, gy2 = min(w, max_x+pad), min(h, max_y+pad)
            
            global_face = frame[gy1:gy2, gx1:gx2]
            if global_face.size > 0:
                global_face = cv2.resize(global_face, (GLOBAL_SIZE, GLOBAL_SIZE))
                cv2.imwrite(os.path.join(OUTPUT_BASE, "GLOBAL", label, video_name.replace(".mp4", ".png")), global_face)

            # Extract AU patches
            for au in FEATURES:
                indices = AU_LANDMARK_MAP.get(au, [])
                if not indices: continue
                
                # Center of the landmarks for this AU
                au_xs = target_row[[f"x_{i}" for i in indices]].values
                au_ys = target_row[[f"y_{i}" for i in indices]].values
                
                cx, cy = int(np.mean(au_xs)), int(np.mean(au_ys))
                
                # Crop PATCH_SIZE x PATCH_SIZE
                r = PATCH_SIZE // 2
                x1, y1 = max(0, cx-r), max(0, cy-r)
                x2, y2 = min(w, cx+r), min(h, cy+r)
                
                patch = frame[y1:y2, x1:x2]
                if patch.size > 0:
                    if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
                        patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
                    
                    out_path = os.path.join(OUTPUT_BASE, au, label, video_name.replace(".mp4", ".png"))
                    cv2.imwrite(out_path, patch)

    print("Extraction complete!")

if __name__ == "__main__":
    extract_patches()

import os
import json
import cv2
import mediapipe as mp
import numpy as np
import requests
from tqdm import tqdm

# ========== Setup ==========
DATA_FILE = "asl/WLASL_v0.3.json"       # your dataset JSON
OUTPUT_DIR = "motions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose


# ========== Helper: Download video ==========
def download_video(url, gloss):
    try:
        file_path = f"temp_{gloss}.mp4"
        r = requests.get(url, stream=True, timeout=15)
        if r.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
            return file_path
        else:
            print(f"❌ Failed to download: {url}")
            return None
    except Exception as e:
        print(f"⚠️ Error downloading {url}: {e}")
        return None


# ========== Helper: Extract keypoints ==========
def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    hands = mp_hands.Hands()
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)

        keypoints = {}

        # Pose landmarks
        if results_pose.pose_landmarks:
            keypoints["pose"] = [
                [lm.x, lm.y] for lm in results_pose.pose_landmarks.landmark
            ]

        # Hand landmarks
        if results_hands.multi_hand_landmarks:
            keypoints["hands"] = []
            for hand_landmarks in results_hands.multi_hand_landmarks:
                hand_points = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                keypoints["hands"].append(hand_points)

        frames_data.append(keypoints)

    cap.release()
    return frames_data


# ========== Main: Iterate all glosses ==========
with open(DATA_FILE, "r") as f:
    data = json.load(f)

for entry in tqdm(data, desc="Processing Glosses"):
    gloss = entry["gloss"]
    output_file = os.path.join(OUTPUT_DIR, f"{gloss}.json")

    if os.path.exists(output_file):
        continue  # skip if already processed

    # Try available instances (pick first valid video)
    for instance in entry["instances"]:
        url = instance["url"]
        if "youtube" in url:
            continue  # skip YouTube for now (no direct download)

        video_path = download_video(url, gloss)
        if not video_path:
            continue

        frames_data = extract_keypoints(video_path)
        os.remove(video_path)

        if len(frames_data) > 0:
            with open(output_file, "w") as out:
                json.dump(frames_data, out)
            print(f"✅ Saved: {output_file}")
            break  # done for this gloss


# import os
# import json
# import cv2
# import numpy as np
# import requests
# from tqdm import tqdm
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions

# # ✅ Running mode comes directly from vision now (newer API)
# VisionRunningMode = vision.RunningMode

# # ========== Setup ==========
# DATA_FILE = "tryasl/WLASL_v0.3.json"
# OUTPUT_DIR = "motions"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ✅ Initialize models (GPU delegate)
# pose_options = vision.PoseLandmarkerOptions(
#     base_options=BaseOptions(
#         model_asset_path="models/pose_landmarker_full.task",
#         delegate=BaseOptions.Delegate.GPU,
#     ),
#     running_mode=VisionRunningMode.VIDEO,
# )

# hand_options = vision.HandLandmarkerOptions(
#     base_options=BaseOptions(
#         model_asset_path="models/hand_landmarker.task",
#         delegate=BaseOptions.Delegate.GPU,
#     ),
#     running_mode=VisionRunningMode.VIDEO,
# )

# pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
# hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)


# # ========== Helper: Download video ==========
# def download_video(url, gloss):
#     try:
#         file_path = f"temp_{gloss}.mp4"
#         r = requests.get(url, stream=True, timeout=15)
#         if r.status_code == 200:
#             with open(file_path, "wb") as f:
#                 for chunk in r.iter_content(chunk_size=1024):
#                     f.write(chunk)
#             return file_path
#         else:
#             print(f"❌ Failed to download: {url}")
#             return None
#     except Exception as e:
#         print(f"⚠️ Error downloading {url}: {e}")
#         return None


# # ========== Helper: Extract keypoints ==========
# def extract_keypoints(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames_data = []
#     frame_index = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert OpenCV frame to MediaPipe Image
#         mp_image = mp.Image(
#             image_format=mp.ImageFormat.SRGB,
#             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
#         )

#         # Run pose and hand detection
#         pose_result = pose_landmarker.detect_for_video(mp_image, frame_index)
#         hand_result = hand_landmarker.detect_for_video(mp_image, frame_index)

#         keypoints = {}

#         # Pose landmarks
#         if pose_result.pose_landmarks:
#             keypoints["pose"] = [
#                 [lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]
#             ]

#         # Hand landmarks
#         if hand_result.hand_landmarks:
#             keypoints["hands"] = []
#             for hand_landmarks in hand_result.hand_landmarks:
#                 keypoints["hands"].append([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

#         frames_data.append(keypoints)
#         frame_index += 1

#     cap.release()
#     return frames_data


# # ========== Main Loop ==========
# with open(DATA_FILE, "r") as f:
#     data = json.load(f)

# for entry in tqdm(data, desc="Processing Glosses"):
#     gloss = entry["gloss"]
#     output_file = os.path.join(OUTPUT_DIR, f"{gloss}.json")

#     if os.path.exists(output_file):
#         continue  # Skip if already processed

#     for instance in entry["instances"]:
#         url = instance["url"]
#         if "youtube" in url:
#             continue  # Skip YouTube (no direct link)

#         video_path = download_video(url, gloss)
#         if not video_path:
#             continue

#         frames_data = extract_keypoints(video_path)
#         os.remove(video_path)

#         if frames_data:
#             with open(output_file, "w") as out:
#                 json.dump(frames_data, out)
#             print(f"✅ Saved: {output_file}")
#             break

import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2

# ========== CONFIG ==========
MOTION_DIR = "motions"           # folder containing your gloss JSON files
SEQUENCE = ["computer", "chair", "drink"]  # order of glosses to animate
SAVE_VIDEO = True
OUTPUT_VIDEO = "asl_sequence.mp4"

# ========== Helper: Load motion data ==========
def load_motion(gloss):
    path = os.path.join(MOTION_DIR, f"{gloss}.json")
    if not os.path.exists(path):
        print(f"⚠️ Motion not found: {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)

# ========== Helper: Draw frame ==========
def draw_frame(ax, keypoints):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # flip Y-axis for correct orientation
    ax.axis('off')

    # Draw pose (body)
    if "pose" in keypoints:
        pose_points = np.array(keypoints["pose"])
        ax.scatter(pose_points[:, 0], pose_points[:, 1], color="skyblue", s=15)

    # Draw hands
    if "hands" in keypoints:
        for hand in keypoints["hands"]:
            hand_points = np.array(hand)
            ax.scatter(hand_points[:, 0], hand_points[:, 1], color="orange", s=10)
            ax.plot(hand_points[:, 0], hand_points[:, 1], color="orange", linewidth=1.0)

# ========== Combine all motion data ==========
def combine_sequence(sequence):
    combined = []
    for gloss in sequence:
        frames = load_motion(gloss)
        combined.extend(frames)
    return combined

# ========== Main Animation ==========
def animate_sequence(sequence):
    all_frames = combine_sequence(sequence)

    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_idx):
        if frame_idx < len(all_frames):
            draw_frame(ax, all_frames[frame_idx])

    ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=40, repeat=False)
    
    if SAVE_VIDEO:
        writer = animation.FFMpegWriter(fps=25)
        ani.save(OUTPUT_VIDEO, writer=writer)
        print(f"🎥 Saved animation as {OUTPUT_VIDEO}")
    else:
        plt.show()

# ========== Run ==========
if __name__ == "__main__":
    print("🦾 Animating ASL sequence:", " → ".join(SEQUENCE))
    animate_sequence(SEQUENCE)

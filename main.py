import glob
import os

import numpy as np
import cv2 as cv

from image_extraction import detect_horizon
from calculations import Velocity_Vector, Accuracy_Check
from results import write_results

IMAGE_DIR = 'MaSTr1325_images_512x384'
MASK_DIR = 'MaSTr1325_masks_512x384'
OUTPUT_PATH = 'horizon_results.xlsx'

# Camera setup - set VFOV_DEG to your camera's actual vertical field of view
VFOV_DEG = 61.0
CAMERA_HEIGHT_FT = 8.0
MIN_CONFIDENCE = 0.75


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    rows = []
    failed_frames = []

    for i, image_path in enumerate(image_paths, 1):
        name = os.path.splitext(os.path.basename(image_path))[0]
        img = cv.imread(image_path)
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        # Step 1 - Horizon detection
        slope, confidence, offset_px, pitch_deg = detect_horizon(img, VFOV_DEG, CAMERA_HEIGHT_FT)
        roll_deg = float(np.degrees(np.arctan(slope)))

        # Step 2 - Velocity offset, only for frames that pass the confidence test
        velocity = Velocity_Vector(roll_deg, pitch_deg, confidence, MIN_CONFIDENCE)
        if velocity is None:
            failed_frames.append(name)

        # Step 3 - Accuracy against the ground-truth mask
        mask_path = os.path.join(MASK_DIR, name + 'm.png')
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        accuracy = None
        if mask is None:
            print(f"No mask for {name}")
        else:
            accuracy = Accuracy_Check(mask, slope, offset_px, pitch_deg, VFOV_DEG, CAMERA_HEIGHT_FT)

        rows.append({
            'image': name,
            'confidence': confidence,
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'offset_px': offset_px,
            'velocity': velocity,
            'accuracy': accuracy,
        })

        if i % 100 == 0 or i == len(image_paths):
            print(f"Processed {i}/{len(image_paths)} - {len(failed_frames)} failed confidence")

    write_results(rows, failed_frames, OUTPUT_PATH, MIN_CONFIDENCE)
    print(f"Frames failing confidence: {len(failed_frames)} of {len(rows)}")
    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

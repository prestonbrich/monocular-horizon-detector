import numpy as np
import cv2 as cv
from sklearn.linear_model import RANSACRegressor

from image_extraction import horizon_offset_pitch

# MaSTr1325 mask labels
OBSTACLE, WATER, SKY, UNKNOWN = 0, 1, 2, 4


def Accuracy_Check(mask, slope, offset_px, pitch_deg, vfov_deg, camera_height_ft=8.0, min_columns=64):
    """Compare a horizon reading against the top edge of the water in a MaSTr1325 mask.

    The top of the water is the horizon whether it meets sky or land/obstacles (shoreline).

    Args:
        mask: ground-truth mask (H x W array of labels 0/1/2/4).
        slope, offset_px, pitch_deg: the readings from detect_horizon.
        vfov_deg, camera_height_ft: same camera settings used for the readings.
        min_columns: minimum number of columns agreeing on the water edge needed to fit a line.

    Returns:
        dict of ground-truth values and errors (measured - truth), or None if the mask
        has no usable water edge.
    """
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    height, width = mask.shape

    # Edge points: per column, the top of the water - the boundary between the first water
    # pixel and the labelled pixel (sky or land/obstacle) above it. A thin unknown band
    # between them is bridged by taking the midpoint.
    xs, ys = [], []
    for x in range(width):
        col = mask[:, x]
        water = np.where(col == WATER)[0]
        if len(water) == 0 or water[0] == 0:
            continue
        y_water = water[0]
        labelled_above = np.where(col[:y_water] != UNKNOWN)[0]
        if len(labelled_above) == 0:
            continue
        y_above = labelled_above.max()
        xs.append(x)
        ys.append((y_above + y_water) / 2)

    if len(xs) < min_columns:
        return None

    # Robust fit: columns where a nearby boat/obstacle hides the horizon give edge points
    # well below the true line, so RANSAC keeps only the columns that agree
    xs, ys = np.array(xs), np.array(ys)
    ransac = RANSACRegressor(residual_threshold=3, random_state=42)
    ransac.fit(xs.reshape(-1, 1), ys)
    inliers = ransac.inlier_mask_
    if inliers.sum() < min_columns:
        return None

    # Fit the ground-truth horizon line and run it through the same offset/pitch maths
    gt_slope, gt_intercept = np.polyfit(xs[inliers], ys[inliers], 1)
    gt_offset_px, gt_pitch_deg = horizon_offset_pitch(gt_slope, gt_intercept, width, height,
                                                      vfov_deg, camera_height_ft)

    roll_deg = np.degrees(np.arctan(slope))
    gt_roll_deg = np.degrees(np.arctan(gt_slope))

    return {
        "gt_slope": float(gt_slope),
        "gt_roll_deg": float(gt_roll_deg),
        "gt_offset_px": gt_offset_px,
        "gt_pitch_deg": gt_pitch_deg,
        "slope_error": float(slope - gt_slope),
        "roll_error_deg": float(roll_deg - gt_roll_deg),
        "offset_error_px": float(offset_px - gt_offset_px),
        "pitch_error_deg": float(pitch_deg - gt_pitch_deg),
        "edge_columns": int(inliers.sum()),
    }


# Calculate Velocity Vector

def Velocity_Vector(roll, pitch, confidence, min_confidence=0.75):
    """Return the (vx, vy, vz) velocity offset, or None if the frame fails the confidence test."""
    if confidence < min_confidence:
        return None

    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)

    # Calculate components (Z-up convention)
    vx = np.cos(pitch_rad)
    vy = np.sin(roll_rad) * np.sin(pitch_rad)
    vz = np.cos(roll_rad) * np.sin(pitch_rad)

    return (float(vx), float(vy), float(vz))
    


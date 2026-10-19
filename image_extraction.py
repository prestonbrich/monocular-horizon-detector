import numpy as np
import cv2 as cv
from sklearn.linear_model import RANSACRegressor

EARTH_RADIUS_M = 6_371_000
FT_TO_M = 0.3048


def detect_horizon(img, vfov_deg, camera_height_ft=8.0, n_strips=32, cell_h=8, margin=8,
                   edge_frac=0.35, search_below_px=48):
    """Find the horizon (or shoreline) line in an image and estimate camera pitch.

    Args:
        img: grayscale or BGR image as a numpy array.
        vfov_deg: the camera's vertical field of view in degrees.
        camera_height_ft: camera height above the water.
        edge_frac: a cell counts as a strong edge if its gradient is at least this
            fraction of the strip's strongest cell.
        search_below_px: how far below the strongest edge to look for the waterline.

    Returns:
        slope: dy/dx of the horizon line in image coordinates (y points down).
            Roll angle in degrees is np.degrees(np.arctan(slope)).
        confidence: fraction of strips that agree with the fitted line (0-1).
            Below ~0.75 the line is often a boat/wake/glare edge rather than the horizon.
        offset_px: how far the detected horizon is from where it would appear at zero
            pitch, measured perpendicular to the line. Positive = detected horizon is lower.
        pitch_deg: camera pitch in degrees. Positive = camera tilted up.
    """
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Step 1 - Determining Maximum Gradient of Vertical Pixels Using Sobel
    blurred = cv.GaussianBlur(img, (11, 11), 0)

    sobel_y = cv.Sobel(blurred, cv.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.abs(sobel_y)

    # Ignore rows near the top/bottom edges, where Sobel padding creates false peaks
    grad_magnitude[:margin] = 0
    grad_magnitude[-margin:] = 0

    # Step 2 - Grid Sampling: split the image into vertical strips and divide each into cells
    height, width = grad_magnitude.shape
    strip_w = width // n_strips
    n_cells = height // cell_h
    points = []

    for i in range(n_strips):
        strip = grad_magnitude[:, i * strip_w:(i + 1) * strip_w]

        # Find the cell with the highest mean gradient
        cells = strip[:n_cells * cell_h].reshape(n_cells, cell_h, -1).mean(axis=(1, 2))
        strongest = np.argmax(cells)

        # On coastlines the strongest edge is often the skyline (top of the land), but the
        # waterline below it is the better pitch reference. Take the lowest strong cell
        # within search_below_px of the strongest - far enough to reach the waterline,
        # not so far that foreground wave edges get picked up.
        limit = min(n_cells, strongest + search_below_px // cell_h + 1)
        best_cell = np.where(cells[:limit] >= edge_frac * cells[strongest])[0].max()

        # Refine to a single row around that cell
        profile = strip.mean(axis=1)
        y0 = max(0, (best_cell - 1) * cell_h)
        y1 = min(height, (best_cell + 2) * cell_h)
        y = y0 + np.argmax(profile[y0:y1])

        points.append([i * strip_w + strip_w // 2, y])

    points = np.array(points)

    # Step 3 - Fitting Line Using RANSAC
    ransac = RANSACRegressor(residual_threshold=15, random_state=42)
    ransac.fit(points[:, 0].reshape(-1, 1), points[:, 1])

    slope = float(ransac.estimator_.coef_[0])
    intercept = float(ransac.estimator_.intercept_)
    confidence = float(ransac.inlier_mask_.mean())

    # Step 4 - Pitch from the horizon's vertical position
    offset_px, pitch_deg = horizon_offset_pitch(slope, intercept, width, height, vfov_deg, camera_height_ft)

    return slope, confidence, offset_px, pitch_deg


def horizon_offset_pitch(slope, intercept, width, height, vfov_deg, camera_height_ft=8.0):
    """Convert a horizon line y = slope * x + intercept into (offset_px, pitch_deg)."""
    # Focal length in pixels from the vertical field of view
    focal_px = (height / 2) / np.tan(np.radians(vfov_deg) / 2)

    # Perpendicular distance from the image centre to the horizon line (positive = line below centre)
    cx, cy = width / 2, height / 2
    dist_px = (slope * cx + intercept - cy) / np.sqrt(1 + slope ** 2)

    # From height h the true horizon sits slightly below level (the "dip"), not exactly at eye level
    h_m = camera_height_ft * FT_TO_M
    dip_rad = np.arccos(EARTH_RADIUS_M / (EARTH_RADIUS_M + h_m))

    # Where the horizon would appear at zero pitch, and how far the detection is from there
    expected_px = focal_px * np.tan(dip_rad)
    offset_px = float(dist_px - expected_px)

    # Horizon seen below the optical axis by angle theta means the camera is tilted up by theta - dip
    pitch_deg = float(np.degrees(np.arctan(dist_px / focal_px) - dip_rad))

    return offset_px, pitch_deg


if __name__ == "__main__":
    image_path = 'MaSTr1325_images_512x384/0001.jpg'
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Placeholder - set this to your camera's actual vertical field of view
    vfov_deg = 61.0

    slope, confidence, offset_px, pitch_deg = detect_horizon(img, vfov_deg, camera_height_ft=8.0)
    print(f"Slope: {slope:.4f}")
    print(f"Frame Roll Angle: {np.degrees(np.arctan(slope)):.2f} degrees")
    print(f"Horizon Offset: {offset_px:.1f} px")
    print(f"Frame Pitch Angle: {pitch_deg:.2f} degrees")
    print(f"Confidence: {confidence:.2f}")


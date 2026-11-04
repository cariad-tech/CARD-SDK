import cv2
import numpy as np


def undistort_bboxes(labels, intrinsics, distortion, output_intrinsics):
    """Return axis-aligned bounds of sampled, undistorted box perimeters."""
    if labels is None:
        return None
    result = []
    fractions = np.linspace(0, 1, 33)
    for label in labels:
        left, top, right, bottom = label["bbox"]
        horizontal = left + fractions * (right - left)
        vertical = top + fractions * (bottom - top)
        perimeter = np.concatenate((
            np.column_stack((horizontal, np.full_like(horizontal, top))),
            np.column_stack((horizontal, np.full_like(horizontal, bottom))),
            np.column_stack((np.full_like(vertical, left), vertical)),
            np.column_stack((np.full_like(vertical, right), vertical)),
        )).astype(np.float64)
        points = cv2.undistortPoints(perimeter[:, None], intrinsics, distortion, P=output_intrinsics)[:, 0]
        if not np.isfinite(points).all():
            raise ValueError("Undistorted bounding-box coordinates must be finite")
        result.append({**label, "bbox": (*points.min(axis=0), *points.max(axis=0))})
    return result
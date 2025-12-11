#!/usr/bin/env python3
"""
Detect a white ping-pong ball in an image using ArUco markers at the four corners
of a black A3 paper as a reference plane. Outputs the ball center coordinates
in the A3 plane (default units: millimeters), and saves an annotated copy of the image.

Requirements:
- opencv-contrib-python
- numpy

Usage (examples):
  python detect_ball_aruco.py path/to/image.jpg
  python detect_ball_aruco.py img.png --out annotated.png --coords-out coords.csv --units mm --aruco-dict 4X4_50

Notes:
- The script expects exactly or at least four ArUco markers to be visible.
- It will infer the A3 plane corners from the outermost corner of each detected marker.
- A3 size is 420 x 297 mm. Mapping will be to (X=0..W, Y=0..H) with origin at the top-left
  of the A3 plane as seen in the image.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2

# ---- Constants ----
ISO_A3_WIDTH_MM = 420.0
ISO_A3_HEIGHT_MM = 297.0

ARUCO_DICT_MAP = {
    # Common options; add more as needed
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

@dataclass
class DetectionResult:
    center_px: Tuple[float, float]
    radius_px: float
    center_plane_mm: Tuple[float, float]
    radius_mm: float
    homography: np.ndarray
    annotated_image: np.ndarray
    a3_corners_px: np.ndarray  # shape (4,2) ordered tl,tr,br,bl


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _get_aruco_detector(aruco_dict_name: str):
    if aruco_dict_name not in ARUCO_DICT_MAP:
        raise ValueError(f"Unknown ArUco dict '{aruco_dict_name}'. Choices: {', '.join(ARUCO_DICT_MAP.keys())}")
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[aruco_dict_name])
    # Compatibility with OpenCV 4.7+ new API vs older API
    try:
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
        return detect
    except Exception:
        # Legacy API fallback
        parameters = cv2.aruco.DetectorParameters_create()
        def detect(gray):
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
            return corners, ids
        return detect


def _select_four_markers(corners_list: List[np.ndarray], ids: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return exactly four markers by picking the four extreme ones if more are found.
    corners_list: list of (1,4,2) arrays; ids: (N,1)
    """
    N = len(corners_list)
    if N < 4:
        raise RuntimeError(f"Expected at least 4 ArUco markers; found {N}")

    centers = np.array([c.reshape(4, 2).mean(axis=0) for c in corners_list])  # (N,2)
    if N == 4:
        return corners_list, ids

    # Select four markers corresponding to extreme corners by sums/diffs
    sums = centers.sum(axis=1)
    diffs = centers[:, 0] - centers[:, 1]
    idx_tl = int(np.argmin(sums))
    idx_br = int(np.argmax(sums))
    idx_tr = int(np.argmax(diffs))
    idx_bl = int(np.argmin(diffs))

    chosen_idxs = []
    for idx in [idx_tl, idx_tr, idx_br, idx_bl]:
        if idx not in chosen_idxs:
            chosen_idxs.append(idx)
    # If duplicates due to colinearity, fill remaining by farthest from centroid
    if len(chosen_idxs) < 4:
        centroid = centers.mean(axis=0)
        dists = np.linalg.norm(centers - centroid, axis=1)
        for idx in np.argsort(-dists):  # descending
            if idx not in chosen_idxs:
                chosen_idxs.append(int(idx))
            if len(chosen_idxs) == 4:
                break

    chosen = [corners_list[i] for i in chosen_idxs[:4]]
    chosen_ids = ids[chosen_idxs[:4]] if ids is not None else None
    return chosen, chosen_ids


def _outermost_corner_points(corners_list: List[np.ndarray]) -> np.ndarray:
    """For each marker, choose the corner point farthest from the global centroid.
    Returns array of shape (4,2) in arbitrary order.
    """
    pts_all = []
    centers = []
    for c in corners_list:
        pts = c.reshape(4, 2)
        pts_all.append(pts)
        centers.append(pts.mean(axis=0))
    centers = np.array(centers)
    global_centroid = centers.mean(axis=0)

    chosen = []
    for pts in pts_all:
        dists = np.linalg.norm(pts - global_centroid, axis=1)
        chosen.append(pts[int(np.argmax(dists))])
    return np.array(chosen, dtype=np.float32)


def _order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order four points into top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    # sort by y
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_two = y_sorted[:2]
    bottom_two = y_sorted[2:]
    # Left-right within rows
    tl, tr = top_two[np.argsort(top_two[:, 0])]
    bl, br = bottom_two[np.argsort(bottom_two[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _compute_homography(corner_pixels_tl_tr_br_bl: np.ndarray,
                        width_mm: float = ISO_A3_WIDTH_MM,
                        height_mm: float = ISO_A3_HEIGHT_MM) -> Tuple[np.ndarray, np.ndarray]:
    dst = np.array([[0.0, 0.0],
                    [width_mm, 0.0],
                    [width_mm, height_mm],
                    [0.0, height_mm]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(corner_pixels_tl_tr_br_bl.astype(np.float32), dst)
    return H, dst


def _perspective_point(H: np.ndarray, point_xy: Tuple[float, float]) -> Tuple[float, float]:
    pts = np.array([[[point_xy[0], point_xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)[0, 0]
    return float(mapped[0]), float(mapped[1])


def _ball_candidates_from_mask(mask: np.ndarray) -> List[Tuple[Tuple[float, float], float, float, float]]:
    """Return list of (center, radius, area, circularity) from white mask contours."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < 30.0:
            continue
        perimeter = float(cv2.arcLength(c, True))
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        (x, y), radius = cv2.minEnclosingCircle(c)
        candidates.append(((float(x), float(y)), float(radius), area, float(circularity)))
    # Sort by best combination: higher circularity, larger area
    candidates.sort(key=lambda t: (t[3], t[2]), reverse=True)
    return candidates


def _detect_ball(image_bgr: np.ndarray,
                 hsv_white_s_max: int = 80,
                 hsv_white_v_min: int = 200,
                 use_hough_fallback: bool = True) -> Tuple[Tuple[float, float], float, np.ndarray]:
    """Detect the ball center and radius in pixel coordinates.
    Returns (center, radius, mask_debug).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, max(0, int(hsv_white_v_min))], dtype=np.uint8)
    upper = np.array([180, max(0, int(hsv_white_s_max)), 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Primary: contour-based
    candidates = _ball_candidates_from_mask(mask)
    for (center, radius, area, circ) in candidates:
        if radius >= 3 and circ >= 0.6:
            return center, radius, mask

    # Fallback: HoughCircles on masked regions
    if use_hough_fallback:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Keep only masked bright regions
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=100, param2=18, minRadius=3, maxRadius=0)
        if circles is not None and len(circles) > 0:
            circles = np.round(circles[0, :]).astype(int)
            # Choose the circle with the highest masked intensity sum
            best = None
            best_score = -1
            for (x, y, r) in circles:
                x, y, r = int(x), int(y), int(r)
                if r < 3:
                    continue
                # Simple score: local mask mean
                x0, y0 = max(0, x - r), max(0, y - r)
                x1, y1 = min(mask.shape[1], x + r), min(mask.shape[0], y + r)
                roi = mask[y0:y1, x0:x1]
                score = float(roi.mean()) if roi.size > 0 else 0.0
                if score > best_score:
                    best_score = score
                    best = (float(x), float(y), float(r))
            if best is not None:
                return (best[0], best[1]), best[2], mask

    raise RuntimeError("Failed to detect the ping-pong ball.")


def detect_ball_and_project(
    image_bgr: np.ndarray,
    aruco_dict_name: str = "4X4_50",
    a3_width_mm: float = ISO_A3_WIDTH_MM,
    a3_height_mm: float = ISO_A3_HEIGHT_MM,
    hsv_white_s_max: int = 80,
    hsv_white_v_min: int = 200
) -> DetectionResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    detect = _get_aruco_detector(aruco_dict_name)
    corners_list, ids = detect(gray)
    if corners_list is None or len(corners_list) < 4:
        raise RuntimeError(f"Expected at least 4 ArUco markers; found {0 if corners_list is None else len(corners_list)}")

    # Select 4 and compute paper corners
    four_corners_list, _ = _select_four_markers(corners_list, ids)
    outer_corners = _outermost_corner_points(four_corners_list)
    ordered_corners = _order_corners_tl_tr_br_bl(outer_corners)

    # Homography from image pixels -> A3 mm plane
    H, _ = _compute_homography(ordered_corners, a3_width_mm, a3_height_mm)

    # Detect ball in pixel space
    center_px, radius_px, mask_debug = _detect_ball(image_bgr, hsv_white_s_max, hsv_white_v_min)

    # Project center (and approximate radius using local mapping)
    cx_mm, cy_mm = _perspective_point(H, center_px)
    # Approximate radius by transforming a point to the right by radius
    r_sample_px = (center_px[0] + radius_px, center_px[1])
    rx_mm, ry_mm = _perspective_point(H, r_sample_px)
    radius_mm = float(np.hypot(rx_mm - cx_mm, ry_mm - cy_mm))

    # Annotate image
    annotated = image_bgr.copy()
    # Draw A3 corners
    for (x, y) in ordered_corners.astype(int):
        cv2.circle(annotated, (int(x), int(y)), 6, (0, 165, 255), thickness=-1)  # orange
    # Draw ball
    cv2.circle(annotated, (int(round(center_px[0])), int(round(center_px[1]))), int(round(radius_px)), (0, 255, 0), 2)
    cv2.drawMarker(annotated, (int(round(center_px[0])), int(round(center_px[1]))), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    label = f"X={cx_mm:.1f} mm, Y={cy_mm:.1f} mm"
    cv2.putText(annotated, label, (int(round(center_px[0] + 10)), int(round(center_px[1] - 10))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return DetectionResult(center_px=center_px,
                           radius_px=radius_px,
                           center_plane_mm=(cx_mm, cy_mm),
                           radius_mm=radius_mm,
                           homography=H,
                           annotated_image=annotated,
                           a3_corners_px=ordered_corners)


def _default_out_paths(input_path: str) -> Tuple[str, str]:
    base, ext = os.path.splitext(input_path)
    return base + "_annotated.png", base + "_coords.csv"


def _write_coords_csv(path: str, x: float, y: float, units: str = "mm") -> None:
    # Write with header if file doesn't exist
    header_needed = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(f"x_{units},y_{units}\n")
        f.write(f"{x:.3f},{y:.3f}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Detect a white ping-pong ball using ArUco markers as A3 plane reference.")
    parser.add_argument("image", help="Path to input image (.png or .jpg)")
    parser.add_argument("--out", dest="out_image", default=None, help="Path to save annotated image (default: <input>_annotated.png)")
    parser.add_argument("--coords-out", dest="coords_out", default=None, help="Path to save coordinates CSV (default: <input>_coords.csv)")
    parser.add_argument("--units", choices=["mm", "cm"], default="mm", help="Units for exported coordinates (default: mm)")
    parser.add_argument("--aruco-dict", dest="aruco_dict", choices=list(ARUCO_DICT_MAP.keys()), default="4X4_50", help="ArUco dictionary to use")
    parser.add_argument("--white-s-max", type=int, default=80, help="HSV S max threshold for white mask (default: 80)")
    parser.add_argument("--white-v-min", type=int, default=200, help="HSV V min threshold for white mask (default: 200)")
    parser.add_argument("--a3-width-mm", type=float, default=ISO_A3_WIDTH_MM, help="A3 width in mm (default: 420)")
    parser.add_argument("--a3-height-mm", type=float, default=ISO_A3_HEIGHT_MM, help="A3 height in mm (default: 297)")

    args = parser.parse_args(argv)

    img_path = args.image
    out_img_path, out_coords_path = _default_out_paths(img_path)
    if args.out_image:
        out_img_path = args.out_image
    if args.coords_out:
        out_coords_path = args.coords_out

    img = _load_image(img_path)

    try:
        result = detect_ball_and_project(
            img,
            aruco_dict_name=args.aruco_dict,
            a3_width_mm=float(args.a3_width_mm),
            a3_height_mm=float(args.a3_height_mm),
            hsv_white_s_max=int(args.white_s_max),
            hsv_white_v_min=int(args.white_v_min),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    x_mm, y_mm = result.center_plane_mm
    units = args.units
    if units == "cm":
        x_out, y_out = x_mm / 10.0, y_mm / 10.0
    else:
        x_out, y_out = x_mm, y_mm

    # Save outputs
    ok_img = cv2.imwrite(out_img_path, result.annotated_image)
    if not ok_img:
        print(f"WARNING: Failed to save annotated image to {out_img_path}", file=sys.stderr)
    try:
        _write_coords_csv(out_coords_path, x_out, y_out, units=units)
    except Exception as e:
        print(f"WARNING: Failed to save coordinates CSV: {e}", file=sys.stderr)

    # Console message
    print(f"Ball center: X={x_out:.2f} {units}, Y={y_out:.2f} {units}")
    print(f"Annotated image: {out_img_path}")
    print(f"Coordinates CSV appended: {out_coords_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

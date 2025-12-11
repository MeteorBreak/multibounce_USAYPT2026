#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
import subprocess
import shutil
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from datetime import datetime

# --------------------------
# Utilities and data classes
# --------------------------

@dataclass
class Impact:
    index: int
    time_sec: float
    frame_index: int
    x_mm: Optional[float] = None
    y_mm: Optional[float] = None
    confidence: float = 0.0
    note: str = ""


def fail(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def check_dep(cmd: str, install_hint: str):
    if shutil.which(cmd) is None:
        fail(f"Required dependency '{cmd}' not found. Install it, e.g.: {install_hint}")


def require_aruco():
    if not hasattr(cv2, "aruco"):
        fail(
            "OpenCV ArUco module not found. Install opencv-contrib-python:\n"
            "  pip install --upgrade opencv-contrib-python"
        )


# --- ArUco compatibility helpers (OpenCV 4.5.x .. 4.10+) ---
def get_aruco_dictionary(aruco_dict_id: int):
    """
    Return a cv2.aruco dictionary instance from a dictionary id enum, handling
    API differences between OpenCV versions.
    """
    aruco = cv2.aruco
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(aruco_dict_id)
    # Older fallback
    if hasattr(aruco, "Dictionary_get"):
        return aruco.Dictionary_get(aruco_dict_id)
    raise RuntimeError("This OpenCV build lacks ArUco dictionary constructors.")


def create_detector_parameters():
    """Create DetectorParameters object across OpenCV versions."""
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()
    if hasattr(aruco, "DetectorParameters"):
        return aruco.DetectorParameters()
    raise RuntimeError("This OpenCV build lacks aruco.DetectorParameters API.")


def detect_aruco_markers(gray: np.ndarray, dictionary):
    """
    Run detectMarkers across OpenCV versions.
    Returns (corners, ids, rejectedCandidates)
    """
    aruco = cv2.aruco
    # Newer API (OpenCV >= 4.7): ArucoDetector class
    if hasattr(aruco, "ArucoDetector"):
        params = create_detector_parameters()
        detector = aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray)
    # Older API
    params = create_detector_parameters()
    return aruco.detectMarkers(gray, dictionary, parameters=params)


def ffmpeg_has_audio(input_path: str) -> bool:
    # Return True if the media has an audio stream (ffprobe)
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            input_path,
        ]
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def decode_audio_to_array(media_path: str, audio_path: Optional[str], target_sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Use ffmpeg to decode audio to mono float32 at target_sr, returning numpy array [-1, 1], length N.
    If audio_path is provided, use it. Otherwise, demux from media_path.
    """
    check_dep("ffmpeg", "sudo apt-get install ffmpeg")
    src = audio_path if audio_path else media_path

    # Build ffmpeg command to output raw float32 PCM to stdout
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or len(proc.stdout) == 0:
        raise RuntimeError(f"ffmpeg failed to decode audio from {src}.\n{proc.stderr.decode(errors='ignore')}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio, target_sr


def short_time_energy(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """
    Short-time energy over frames. x is mono float32.
    """
    n_frames = 1 + max(0, (len(x) - frame_len) // hop)
    if n_frames <= 0:
        return np.array([], dtype=np.float32)
    # Frame with stride trick
    shape = (n_frames, frame_len)
    strides = (x.strides[0]*hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    ste = np.mean(frames * frames, axis=1).astype(np.float32)
    return ste


def detect_impacts_from_audio(
    audio: np.ndarray,
    sr: int,
    min_separation_s: float = 0.25,
    sensitivity: float = 3.0,
    frame_ms: float = 20.0,
    hop_ms: float = 5.0,
) -> List[float]:
    """
    Simple transient detector using short-time energy + adaptive threshold.
    sensitivity: higher => fewer detections (threshold multiplier on MAD)
    Returns list of times (seconds).
    """
    if len(audio) == 0:
        return []
    # Pre-emphasis high-pass-ish: difference filter to emphasize transients
    x = audio.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    x = np.concatenate([[0.0], np.diff(x)])

    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    frame_len = max(frame_len, 128)
    hop = max(hop, 32)

    ste = short_time_energy(x, frame_len, hop)
    if ste.size == 0:
        return []

    # Normalize
    ste = (ste - np.median(ste)) / (np.max(ste) - np.min(ste) + 1e-9)

    # Smoothed
    kernel = np.ones(5, dtype=np.float32) / 5.0
    ste_s = np.convolve(ste, kernel, mode="same")

    # Adaptive threshold using median absolute deviation
    mad = np.median(np.abs(ste_s - np.median(ste_s))) + 1e-9
    thr = np.median(ste_s) + sensitivity * mad

    # Peak picking
    above = ste_s > thr
    # Find rising edges to mark peaks roughly
    candidates = np.where(np.logical_and(above, np.concatenate([[False], ~above[:-1]])))[0].tolist()

    # Refine each candidate to local maximum in a small window
    refined = []
    win = 3
    for c in candidates:
        lo = max(0, c - win)
        hi = min(len(ste_s), c + win + 1)
        peak_idx = lo + int(np.argmax(ste_s[lo:hi]))
        refined.append(peak_idx)

    # Enforce min separation
    min_sep_frames = int((min_separation_s * sr) / hop)
    kept = []
    for p in refined:
        if not kept or (p - kept[-1]) >= min_sep_frames:
            kept.append(p)
        else:
            # keep the stronger one
            if ste_s[p] > ste_s[kept[-1]]:
                kept[-1] = p

    times = [(p * hop) / float(sr) for p in kept]
    return times


def load_frame_at_time(cap: cv2.VideoCapture, fps: float, t_sec: float) -> Tuple[Optional[np.ndarray], int]:
    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None, frame_idx
    return frame, frame_idx


def try_detect_aruco_corners(
    img_bgr: np.ndarray,
    attempt_dicts: List[str],
) -> Optional[np.ndarray]:
    """
    Detects ArUco markers and returns 4 corner points ordered TL, TR, BR, BL
    using marker centers as proxies for sheet corners.
    """
    require_aruco()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detections = []
    for dict_name in attempt_dicts:
        try:
            aruco_dict = getattr(cv2.aruco, dict_name)
        except AttributeError:
            continue
        try:
            dictionary = get_aruco_dictionary(aruco_dict)
        except Exception:
            continue
        try:
            corners, ids, _ = detect_aruco_markers(gray, dictionary)
        except Exception:
            continue
        if ids is None or len(ids) < 4:
            continue
        # Compute centers for each marker
        centers = []
        for c in corners:
            pts = c.reshape(-1, 2)
            centers.append(np.mean(pts, axis=0))
        centers = np.array(centers, dtype=np.float32)
        if centers.shape[0] < 4:
            continue
        # Choose the 4 that form the largest quadrilateral by convex hull area
        hull = cv2.convexHull(centers)
        if len(hull) < 4:
            continue
        # If hull has more than 4 points, approximate polygon
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) != 4:
            # fallback: pick 4 extreme points (min x+y, max x-y etc.)
            pts = centers
        else:
            pts = approx.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] >= 4:
            # If >4 points, reduce to 4 using extreme order
            if pts.shape[0] > 4:
                pts = order_quad_by_position(pts)  # sort/order then take first 4
                pts = pts[:4]
            ordered = order_quad_by_position(pts)
            return ordered.astype(np.float32)
        detections.append(centers)

    # Fallback: try to detect largest quadrilateral (the A3 sheet) by thresholding
    sheet = detect_sheet_by_contour(gray)
    if sheet is not None:
        return order_quad_by_position(sheet.astype(np.float32))

    return None


def order_quad_by_position(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as TL, TR, BR, BL.
    If more than 4 points, it still returns 4 after selecting extremes.
    """
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] > 4:
        # Select extremes
        s = pts.sum(axis=1)
        d = pts[:, 0] - pts[:, 1]
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmax(d)]
        bl = pts[np.argmin(d)]
        ordered = np.vstack([tl, tr, br, bl])
        # Deduplicate in case of overlaps
        _, idx = np.unique(ordered, axis=0, return_index=True)
        ordered = ordered[np.sort(idx)]
        # If still >4, take first 4 unique
        if ordered.shape[0] > 4:
            ordered = ordered[:4]
        if ordered.shape[0] == 4:
            return ordered
        # fallback
        pts = pts[:4]
    # Exactly 4
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.vstack([tl, tr, br, bl]).astype(np.float32)


def detect_sheet_by_contour(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Fallback: find largest dark quadrilateral (the black A3 sheet).
    Returns 4 points if found, else None.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold inverted (sheet dark)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    return approx.reshape(-1, 2).astype(np.float32)


def compute_homography_to_a3(
    img_corners_tl_tr_br_bl: np.ndarray,
    a3_w_mm: float = 420.0,
    a3_h_mm: float = 297.0,
    marker_offset_mm: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns homography H such that x_mm = H * x_img (homogeneous).
    If marker_offset_mm > 0, expands/contract corners along vector from center to approximate sheet corners.
    """
    pts_img = img_corners_tl_tr_br_bl.astype(np.float32)
    # Adjust for marker offset if provided
    if marker_offset_mm != 0.0:
        # Approximate by shifting points outward/inward proportionally in image space.
        c = np.mean(pts_img, axis=0)
        for i in range(4):
            v = pts_img[i] - c
            nv = v / (np.linalg.norm(v) + 1e-6)
            # Assume approximately 1 mm corresponds to k pixels locally; we don't know k, so leave as 0 by default.
            # If user provides offset, we apply a small pixel offset heuristically: 1 mm ~ 1 px as rough fallback.
            pts_img[i] = pts_img[i] + nv * marker_offset_mm  # heuristic
    # Destination: A3 plane in millimeters
    pts_dst = np.array([
        [0.0, 0.0],             # TL
        [a3_w_mm, 0.0],         # TR
        [a3_w_mm, a3_h_mm],     # BR
        [0.0, a3_h_mm],         # BL
    ], dtype=np.float32)
    H, _ = cv2.findHomography(pts_img, pts_dst, method=cv2.RANSAC)
    return H, pts_dst


def warp_to_a3(img_bgr: np.ndarray, H: np.ndarray, a3_w_mm: float, a3_h_mm: float, px_per_mm: float) -> np.ndarray:
    size = (int(round(a3_w_mm * px_per_mm)), int(round(a3_h_mm * px_per_mm)))
    warped = cv2.warpPerspective(img_bgr, H, size, flags=cv2.INTER_LINEAR)
    return warped


def detect_ball_center_in_warp(
    warped_bgr: np.ndarray,
    px_per_mm: float,
    ball_diameter_mm: float = 40.0,
) -> Tuple[Optional[Tuple[float, float]], float, np.ndarray]:
    """
    Detect ball center in the warped image. Returns (x_px, y_px), confidence, overlay_image.
    """
    vis = warped_bgr.copy()
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    # Suppress sheet texture, emphasize bright ball
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # Normalize contrast
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)
    # Threshold high brights (ball is white)
    _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # HoughCircles parameters
    expected_r_px = max(6, int(round((ball_diameter_mm * 0.5) * px_per_mm)))
    min_r = int(max(4, expected_r_px * 0.6))
    max_r = int(expected_r_px * 1.6)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, expected_r_px),
        param1=100,
        param2=20,
        minRadius=min_r,
        maxRadius=max_r,
    )

    best = None
    best_score = -1.0
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        h, w = gray.shape
        for (x, y, r) in circles:
            # Score: brightness at center + circle size closeness
            center_val = int(gray[y, x]) if 0 <= y < h and 0 <= x < w else 0
            size_score = 1.0 - min(1.0, abs(r - expected_r_px) / (expected_r_px + 1e-6))
            score = 0.7 * (center_val / 255.0) + 0.3 * size_score
            if score > best_score:
                best_score = score
                best = (float(x), float(y))
        if best is not None:
            cv2.circle(vis, (int(best[0]), int(best[1])), int(expected_r_px), (0, 255, 0), 2)
            cv2.circle(vis, (int(best[0]), int(best[1])), 2, (0, 0, 255), 3)

    # Fallback: bright blob contour
    if best is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        open_img = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        cnts, _ = cv2.findContours(open_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            areas = [(cv2.contourArea(c), c) for c in cnts]
            areas.sort(key=lambda x: x[0], reverse=True)
            for area, c in areas[:5]:
                if area < math.pi * (min_r ** 2) * 0.3:
                    continue
                (x, y), r = cv2.minEnclosingCircle(c)
                r = float(r)
                if r < min_r or r > max_r:
                    continue
                M = cv2.moments(c)
                if M["m00"] > 1e-6:
                    cx = float(M["m10"] / M["m00"])
                    cy = float(M["m01"] / M["m00"])
                    best = (cx, cy)
                    # confidence heuristic
                    size_score = 1.0 - min(1.0, abs(r - expected_r_px) / (expected_r_px + 1e-6))
                    best_score = 0.5 + 0.5 * size_score
                    cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 2, (0, 0, 255), 3)
                    break

    if best is None:
        return None, 0.0, vis
    return best, float(best_score), vis


def img_points_to_mm(pt_px: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """
    Map image (x, y) in original image to millimeter coordinates using homography H (img->mm).
    """
    x, y = pt_px
    vec = np.array([x, y, 1.0], dtype=np.float64)
    mm = H @ vec
    mm /= (mm[2] + 1e-12)
    return float(mm[0]), float(mm[1])


# --------------------------
# Visualization helpers
# --------------------------

def _ensure_color(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def make_composite_frame(left_bgr: np.ndarray, right_bgr: np.ndarray, left_title: str = "original", right_title: str = "warped") -> np.ndarray:
    """Stack two images side-by-side with simple headers."""
    left = _ensure_color(left_bgr)
    right = _ensure_color(right_bgr)
    h = max(left.shape[0], right.shape[0])
    # Resize to same height
    def resize_h(img, target_h):
        if img.shape[0] == target_h:
            return img
        scale = target_h / img.shape[0]
        new_w = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    left = resize_h(left, h)
    right = resize_h(right, h)

    # Add titles
    def add_title(img, title: str):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
        cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return out

    left = add_title(left, left_title)
    right = add_title(right, right_title)
    composite = np.hstack([left, right])
    # Footer with timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(composite, ts, (10, composite.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return composite


def try_get_video_writer(path: str, frame_size: Tuple[int, int], fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if path.lower().endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, max(1.0, fps), frame_size)
    if not writer.isOpened():
        return None
    return writer


def plot_impacts(impacts: List[Impact], a3_w_mm: float, a3_h_mm: float, save_path: Optional[str] = None, show: bool = True):
    # Lazy import to avoid hard dependency if not used
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Ellipse
    except Exception as e:
        print(f"Plotting skipped: matplotlib not available ({e}).", file=sys.stderr)
        return

    xs = [e.x_mm for e in impacts if e.x_mm is not None]
    ys = [e.y_mm for e in impacts if e.y_mm is not None]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)
    # A3 rectangle
    ax.add_patch(Rectangle((0, 0), a3_w_mm, a3_h_mm, fill=False, edgecolor='black', linewidth=1.5))

    if xs and ys:
        ax.scatter(xs, ys, s=40, c='tab:blue', label='impacts', alpha=0.85)
        # Mean
        mx, my = float(np.mean(xs)), float(np.mean(ys))
        ax.scatter([mx], [my], s=80, c='tab:red', marker='x', label='mean')
        # Median
        mdx, mdy = float(np.median(xs)), float(np.median(ys))
        ax.scatter([mdx], [mdy], s=60, c='tab:orange', marker='+', label='median')

        # Covariance ellipse (1 std dev)
        if len(xs) >= 2:
            data = np.vstack([xs, ys])
            cov = np.cov(data)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1])
            ell = Ellipse((mx, my), width, height, angle=theta, edgecolor='tab:green', facecolor='none', linestyle='--', linewidth=1.2, label='1σ ellipse')
            ax.add_patch(ell)

        ax.legend(loc='upper right')

    ax.set_xlim(-10, a3_w_mm + 10)
    ax.set_ylim(-10, a3_h_mm + 10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', linewidth=0.6)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Ping-pong ball impacts on A3 (mm)')

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches='tight')
            print(f"Saved impact plot to {save_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}", file=sys.stderr)

    if show:
        try:
            plt.show()
        except Exception:
            pass
    else:
        plt.close(fig)


# --------------------------
# Main pipeline
# --------------------------

def process(
    video_path: str,
    audio_path: Optional[str],
    out_csv: str,
    out_debug_dir: Optional[str],
    viz_show: bool,
    viz_delay_ms: int,
    viz_video_path: Optional[str],
    a3_w_mm: float,
    a3_h_mm: float,
    px_per_mm: float,
    marker_offset_mm: float,
    ball_diameter_mm: float,
    min_separation_s: float,
    sensitivity: float,
    seek_pad_frames: int,
    aruco_dicts: List[str],
) -> List[Impact]:
    if not os.path.exists(video_path):
        fail(f"Video file not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        fail(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Decode audio
    if audio_path is None:
        if not ffmpeg_has_audio(video_path):
            fail("No audio stream found in the video. Provide --audio path to an audio file (wav/m4a).")
    audio, sr = decode_audio_to_array(video_path, audio_path, target_sr=48000)

    # Detect impact times
    times = detect_impacts_from_audio(
        audio,
        sr,
        min_separation_s=min_separation_s,
        sensitivity=sensitivity,
        frame_ms=20.0,
        hop_ms=5.0,
    )
    if not times:
        print("No impact sounds detected.", file=sys.stderr)
        return []

    if out_debug_dir:
        os.makedirs(out_debug_dir, exist_ok=True)

    # Setup visualization video writer if requested
    writer = None
    composite_size = None

    impacts: List[Impact] = []
    for idx, t in enumerate(times):
        # Load target frame (and optionally search within a small neighborhood if needed)
        best_frame = None
        best_frame_idx = None
        # Optionally scan a small window around the estimated frame to maximize marker detectability
        base_fi = int(round(t * fps))
        scan_offsets = [0]
        if seek_pad_frames > 0:
            scan_offsets = list(range(-seek_pad_frames, seek_pad_frames + 1))

        sheet_corners = None
        H_img_to_mm = None
        frame_used = None

        for off in scan_offsets:
            fi = int(np.clip(base_fi + off, 0, max(0, total_frames - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            corners = try_detect_aruco_corners(frame, aruco_dicts)
            if corners is None or corners.shape[0] != 4:
                continue
            H_img_to_mm, _ = compute_homography_to_a3(corners, a3_w_mm, a3_h_mm, marker_offset_mm)
            sheet_corners = corners
            best_frame = frame
            best_frame_idx = fi
            frame_used = frame.copy()
            break

        if best_frame is None or H_img_to_mm is None:
            impacts.append(Impact(index=idx, time_sec=t, frame_index=base_fi, x_mm=None, y_mm=None,
                                  confidence=0.0, note="Sheet not found"))
            continue

        # Warp to A3 plane
        warped = warp_to_a3(best_frame, H_img_to_mm, a3_w_mm, a3_h_mm, px_per_mm)

        # Detect ball center in the warped image
        ball_px, conf, vis = detect_ball_center_in_warp(warped, px_per_mm, ball_diameter_mm=ball_diameter_mm)
        if ball_px is None:
            impacts.append(Impact(index=idx, time_sec=t, frame_index=best_frame_idx, x_mm=None, y_mm=None,
                                  confidence=0.0, note="Ball not found"))
        else:
            x_mm = ball_px[0] / px_per_mm
            y_mm = ball_px[1] / px_per_mm
            impacts.append(Impact(index=idx, time_sec=t, frame_index=best_frame_idx, x_mm=x_mm, y_mm=y_mm,
                                  confidence=conf, note=""))

        # Debug output and visualization
        if out_debug_dir:
            # Draw markers on original
            dbg = frame_used.copy()
            for p in sheet_corners.astype(int):
                cv2.circle(dbg, tuple(p), 6, (0, 0, 255), -1)
            cv2.putText(dbg, f"t={t:.3f}s f={best_frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out_debug_dir, f"impact_{idx:03d}_frame.jpg"), dbg)
            cv2.imwrite(os.path.join(out_debug_dir, f"impact_{idx:03d}_warp.jpg"), vis)

        # Build composite frame (original annotated + warped visualization)
        dbg_local = frame_used.copy()
        for p in sheet_corners.astype(int):
            cv2.circle(dbg_local, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(dbg_local, f"t={t:.3f}s f={best_frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        comp = make_composite_frame(dbg_local, vis, left_title="original", right_title="warped to A3")

        # Show window if requested
        if viz_show:
            try:
                cv2.imshow("Impact detection", comp)
                key = cv2.waitKey(max(1, viz_delay_ms)) & 0xFF
                if key == 27:  # ESC to abort early
                    viz_show = False
            except Exception:
                pass

        # Save visualization video if requested
        if viz_video_path:
            if writer is None:
                composite_size = (comp.shape[1], comp.shape[0])
                writer = try_get_video_writer(viz_video_path, composite_size, fps=fps)
                if writer is None:
                    print(f"Failed to open video writer at {viz_video_path}", file=sys.stderr)
                    viz_video_path = None
            if writer is not None:
                writer.write(comp)

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_index", "time_sec", "frame_index", "x_mm", "y_mm", "confidence", "note"])
        for e in impacts:
            w.writerow([e.index, f"{e.time_sec:.6f}", e.frame_index,
                        "" if e.x_mm is None else f"{e.x_mm:.3f}",
                        "" if e.y_mm is None else f"{e.y_mm:.3f}",
                        f"{e.confidence:.3f}", e.note])

    # Cleanup visualization resources
    if writer is not None:
        writer.release()
    try:
        if viz_show:
            cv2.destroyAllWindows()
    except Exception:
        pass

    return impacts


def parse_args():
    p = argparse.ArgumentParser(
        description="Detect ping-pong ball landing points on an A3 sheet using audio impacts and ArUco markers."
    )
    p.add_argument("--video", required=True, help="Path to the video file (e.g., mp4, mov).")
    p.add_argument("--audio", default=None, help="Optional separate audio file (wav/m4a). If not set, audio is taken from the video.")
    p.add_argument("--out", default="impacts.csv", help="Output CSV file.")
    p.add_argument("--debug-dir", default=None, help="Optional directory to save debug images.")
    # Visualization options
    p.add_argument("--show", action="store_true", help="Show a live window with detection overlays for each impact.")
    p.add_argument("--show-delay-ms", type=int, default=700, help="Delay per impact frame when --show is used.")
    p.add_argument("--save-video", default=None, help="Optional path to save a visualization video (e.g., impacts_viz.mp4).")
    # Plotting options
    p.add_argument("--plot", action="store_true", help="Show a scatter plot of detected impacts with mean/median.")
    p.add_argument("--plot-file", default=None, help="Optional path to save the plot image (e.g., impacts_plot.png).")
    p.add_argument("--px-per-mm", type=float, default=3.0, help="Warp resolution in pixels per millimeter.")
    p.add_argument("--a3-w-mm", type=float, default=420.0, help="A3 width in millimeters.")
    p.add_argument("--a3-h-mm", type=float, default=297.0, help="A3 height in millimeters.")
    p.add_argument("--marker-offset-mm", type=float, default=0.0, help="Approximate distance from marker centers to actual sheet corners (mm).")
    p.add_argument("--ball-diameter-mm", type=float, default=40.0, help="Ping-pong ball diameter in millimeters.")
    p.add_argument("--min-impact-gap", type=float, default=0.25, help="Minimum separation between detected impacts (seconds).")
    p.add_argument("--sensitivity", type=float, default=3.0, help="Audio detection sensitivity (higher=fewer detections).")
    p.add_argument("--seek-pad-frames", type=int, default=0, help="Search ±N frames around the impact time for better marker detection.")
    p.add_argument("--aruco-dicts", default="DICT_4X4_50,DICT_5X5_100,DICT_6X6_250,DICT_APRILTAG_36h11",
                   help="Comma-separated OpenCV ArUco dictionary names to try.")
    return p.parse_args()


def main():
    args = parse_args()
    aruco_dicts = [s.strip() for s in args.aruco_dicts.split(",") if s.strip()]
    try:
        impacts = process(
            video_path=args.video,
            audio_path=args.audio,
            out_csv=args.out,
            out_debug_dir=args.debug_dir,
            viz_show=args.show,
            viz_delay_ms=args.show_delay_ms,
            viz_video_path=args.save_video,
            a3_w_mm=args.a3_w_mm,
            a3_h_mm=args.a3_h_mm,
            px_per_mm=args.px_per_mm,
            marker_offset_mm=args.marker_offset_mm,
            ball_diameter_mm=args.ball_diameter_mm,
            min_separation_s=args.min_impact_gap,
            sensitivity=args.sensitivity,
            seek_pad_frames=args.seek_pad_frames,
            aruco_dicts=aruco_dicts,
        )
        print(f"Wrote {len(impacts)} impact(s) to {args.out}")
        if args.debug_dir:
            print(f"Debug images saved to {args.debug_dir}")
        # Plotting after processing
        if args.plot or args.plot_file:
            plot_impacts(impacts, args.a3_w_mm, args.a3_h_mm, save_path=args.plot_file, show=args.plot)
    except Exception as e:
        fail(str(e))


if __name__ == "__main__":
    main()
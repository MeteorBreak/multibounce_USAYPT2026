import cv2
import numpy as np
import soundfile as sf
import tempfile
import subprocess
from pathlib import Path
from scipy.signal import find_peaks

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
A3_WIDTH_MM = 420.0
A3_HEIGHT_MM = 297.0

if not hasattr(cv2, "aruco"):
    raise ImportError("The cv2.aruco module is unavailable. Install opencv-contrib-python to enable ArUco support.")

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
try:
    ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT)
except AttributeError:
    ARUCO_DETECTOR = None

def extract_audio(video_path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", str(sr), tmp_name],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    audio, sr = sf.read(tmp_name, dtype="float32")
    Path(tmp_name).unlink(missing_ok=True)
    return audio, sr

def _order_quadrilateral(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered

def find_aruco_corners(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ARUCO_DETECTOR is not None:
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
    if ids is None or len(ids) < 4:
        return None
    quads = [corner[0] for corner in corners]
    if len(quads) > 4:
        areas = [abs(cv2.contourArea(q)) for q in quads]
        top_indices = np.argsort(areas)[-4:]
        quads = [quads[i] for i in top_indices]
    combined = np.vstack(quads).astype(np.float32)
    hull = cv2.convexHull(combined)
    if hull.shape[0] < 4:
        return None
    if hull.shape[0] > 4:
        hull = cv2.approxPolyDP(hull, epsilon=5.0, closed=True)
    if hull.shape[0] != 4:
        return None
    return _order_quadrilateral(hull.reshape(-1, 2))

def compute_homography(paper_corners: np.ndarray) -> np.ndarray:
    dst = np.array(
        [[0, 0], [A3_WIDTH_MM, 0], [A3_WIDTH_MM, A3_HEIGHT_MM], [0, A3_HEIGHT_MM]], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(paper_corners, dst)

def detect_ball_center(frame: np.ndarray) -> tuple[float, float] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=80, param2=18, minRadius=5, maxRadius=30)
    if circles is None:
        return None
    circles = np.round(circles[0, :]).astype("int")
    return tuple(circles[0][:2])


def detect_impacts(audio: np.ndarray, sr: int, fps: float, sensitivity: float = 3.0) -> list[int]:
    window = max(1, int(sr * 0.01))
    energy = np.convolve(np.abs(audio), np.ones(window) / window, mode="same")
    height = np.mean(energy) * sensitivity
    peaks, _ = find_peaks(energy, height=height, distance=int(sr * 0.1))
    times = peaks / sr
    return np.unique(np.round(times * fps).astype(int)).tolist()

def main(video_path: str) -> list[dict[str, float]]:
    video = cv2.VideoCapture(video_path)
    assert video.isOpened(), "Cannot open video."
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    audio, sr = extract_audio(Path(video_path))
    impact_frames = set(detect_impacts(audio, sr, fps))
    results = []
    H = None

    probe_limit = int(max(30, fps * 10))
    for _ in range(probe_limit):
        ret, probe_frame = video.read()
        if not ret:
            break
        corners = find_aruco_corners(probe_frame)
        if corners is not None:
            H = compute_homography(corners)
            break
    if H is None:
        video.release()
        raise RuntimeError("Failed to detect the ArUco board in any frame.")
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        frame_idx = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        if not ret:
            break
        if frame_idx in impact_frames:
            center = detect_ball_center(frame)
            if center:
                px = np.array([[center[0], center[1], 1.0]], dtype=np.float32).T
                warped = H @ px
                warped /= warped[2, 0]
                results.append({"frame": frame_idx, "x_mm": warped[0, 0], "y_mm": warped[1, 0]})
    video.release()
    return results

if __name__ == "__main__":
    import json, sys
    coords = main(sys.argv[1])
    print(json.dumps(coords, indent=2))
# multibounce: plotting CSV points

This repo contains three CSV files with 32 pairs of x-y coordinates each under `25_10_23/` and a small Python script to plot them together.

## How to run

- Requires Python 3.8+ and `matplotlib`.
- The script uses a headless backend, so it will save an image instead of popping up a window.

Run from the repo root:

```fish
# (Optional) install matplotlib if you don't have it yet
pip install matplotlib

# Generate the plot (reads 25_10_23/b_1.csv .. b_3.csv)
python plot_points.py

# Or specify a custom data directory (must contain b_1.csv, b_2.csv, b_3.csv)
python plot_points.py 25_10_23
```

The output image will be written to `25_10_23/combined_points.png`.

## Notes

- Each file's points are plotted with a distinct color and labeled in the legend.
- The script is tolerant of a single header row; non-numeric lines are skipped with a warning.

## Aligned central tendency plot

To visualize the central tendency (shape/spread) without absolute offsets, use the aligned variant:

```fish
# Mean-centered overlay (default)
python plot_points_aligned.py

# Median-centered overlay
python plot_points_aligned.py --method median
```

Output: `25_10_23/combined_points_aligned.png`.

## ArUco-based ball detection (A3 plane)

A separate script detects four ArUco markers placed at the corners of a black A3 paper, builds the A3 plane, detects a white ping-pong ball, and exports its center coordinates on that plane.

### Install dependencies

This part requires OpenCV with the ArUco module:

```fish
pip install -r requirements.txt
```

### Run the detector

```fish
# Basic usage
python detect_ball_aruco.py path/to/image.jpg

# With custom outputs and options
python detect_ball_aruco.py path/to/image.png \
	--out path/to/annotated.png \
	--coords-out path/to/coords.csv \
	--units mm \
	--aruco-dict 4X4_50 \
	--white-s-max 80 \
	--white-v-min 200
```

### What it does

- Detects ≥4 ArUco markers. If more are found, it picks the four extreme ones to form the A3 corners.
- Computes a homography mapping image pixels to the A3 plane (default 420 x 297 mm, origin at the top-left corner).
- Detects the white ball (HSV-based mask with a circle/circularity check; HoughCircles fallback).
- Projects the ball center to A3 coordinates and saves:
	- An annotated image with detected A3 corners and ball center.
	- A CSV file with the ball center coordinates.

### Outputs

- If you don't pass `--out` or `--coords-out`, the script writes next to the input image:
	- `<input>_annotated.png`
	- `<input>_coords.csv` (appended; header added if new)

### Options

- `--units {mm,cm}`: output coordinates in millimeters (default) or centimeters.
- `--aruco-dict`: choose a predefined ArUco dictionary (e.g., `4X4_50`, `5X5_100`, `ARUCO_ORIGINAL`).
- `--white-s-max` and `--white-v-min`: adjust HSV thresholds if lighting varies.
- `--a3-width-mm`, `--a3-height-mm`: override A3 size if needed.

### Tips

- Ensure all 4 markers are visible and near each paper corner.
- Avoid strong glare on the ball; adjust `--white-s-max` / `--white-v-min` for your lighting.
- Use an ArUco dictionary that matches your printed markers.

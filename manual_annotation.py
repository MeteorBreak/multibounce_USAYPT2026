import cv2
import numpy as np
import argparse
import sys
import os

def order_points(pts):
    """
    Sorts calibration points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-left
    rect[2] = pts[np.argmax(s)]   # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    return rect

class VideoAnnotator:
    def __init__(self, video_path, ref_length, ref_width):
        self.video_path = video_path
        self.ref_length = float(ref_length)
        self.ref_width = float(ref_width)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            sys.exit(1)
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        
        self.window_name = "Annotation Tool"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Frame", self.window_name, 0, max(1, self.total_frames - 1), self.on_trackbar)
        
        self.calib_points = []
        self.object_point = None
        self.frame_buffer = None
        self.homography_matrix = None
        
        # Load first frame for calibration
        self.update_frame()
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def on_trackbar(self, val):
        self.current_frame_idx = val
        self.update_frame()
        self.draw()

    def update_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.frame_buffer = frame
        else:
            print(f"Error reading frame {self.current_frame_idx}")

    def on_mouse(self, event, x, y, flags, param):
        if self.frame_buffer is not None:
            h, w = self.frame_buffer.shape[:2]
            if y >= h:
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Object Selection (Ctrl + Left)
                self.object_point = (x, y)
            else:
                # Calibration: Add point if less than 4
                if len(self.calib_points) < 4:
                    self.calib_points.append((x, y))
                    if len(self.calib_points) == 4:
                        self.compute_homography()
                else:
                     # Optional: Logic to reset or refine points could go here
                     pass
            self.draw()

    def compute_homography(self):
        if len(self.calib_points) != 4:
            return

        # Image coordinates
        pts_src = np.array(self.calib_points, dtype="float32")
        pts_src = order_points(pts_src)

        # Physical coordinates (mm)
        # Assuming Center is (0,0)
        # Order: TL, TR, BR, BL relative to center
        # X axis: Right, Y axis: Down (to match image convention) in terms of layout
        half_l = self.ref_length / 2.0
        half_w = self.ref_width / 2.0
        
        # Define physical target points in the same order (TL, TR, BR, BL)
        # Note: If image Y is Down, then TL is (-x, -y), TR is (+x, -y), etc.
        # But logically, let's treat it as a standard 2D plane.
        # TL: (-L/2, -W/2)
        # TR: ( L/2, -W/2)
        # BR: ( L/2,  W/2)
        # BL: (-L/2,  W/2)
        
        pts_dst = np.array([
            [-half_l, -half_w], # TL
            [ half_l, -half_w], # TR
            [ half_l,  half_w], # BR
            [-half_l,  half_w]  # BL
        ], dtype="float32")

        self.homography_matrix, status = cv2.findHomography(pts_src, pts_dst)
        print("Calibration completed (Homography Matrix calculated).")

    def calculate_position(self):
        if self.homography_matrix is None:
            print("Error: Reference rectangle not defined completely (need 4 points).")
            return None
        if self.object_point is None:
            print("Error: Object position not selected (Right Click).")
            return None

        # Transform object point
        pt = np.array([[self.object_point]], dtype="float32")
        pt_transformed = cv2.perspectiveTransform(pt, self.homography_matrix)
        
        return pt_transformed[0][0]

    def draw(self):
        if self.frame_buffer is None:
            return
            
        h, w = self.frame_buffer.shape[:2]
        ui_height = 100
        
        # Create canvas with extra space at bottom for UI to avoid covering video
        canvas = np.zeros((h + ui_height, w, 3), dtype="uint8")
        canvas[:h, :w] = self.frame_buffer
        
        display_img = canvas
        
        # Draw calibration points
        for i, pt in enumerate(self.calib_points):
            cv2.circle(display_img, pt, 5, (0, 255, 0), -1)
            cv2.putText(display_img, str(i+1), (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(self.calib_points) == 4:
             sorted_pts = order_points(np.array(self.calib_points))
             cv2.polylines(display_img, [np.int32(sorted_pts)], True, (0, 255, 0), 2)

        # Draw object point
        if self.object_point:
            cv2.circle(display_img, self.object_point, 5, (0, 0, 255), -1)
            cv2.line(display_img, (self.object_point[0]-10, self.object_point[1]), (self.object_point[0]+10, self.object_point[1]), (0,0,255), 1)
            cv2.line(display_img, (self.object_point[0], self.object_point[1]-10), (self.object_point[0], self.object_point[1]+10), (0,0,255), 1)

        # Visual UI - Draw in the padded bottom area
        ui_y_start = h + 25
        line_height = 25
        
        info_calib = f"Calibration: {len(self.calib_points)}/4 points"
        info_frame = f"Frame: {self.current_frame_idx}/{self.total_frames}"
        cv2.putText(display_img, f"{info_calib} | {info_frame}", (10, ui_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_img, "Mouse: Left=Calib Corner, Ctrl+Left=Object", (10, ui_y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_img, "Keys: Left/Right=Prev/Next Frame | Enter=Confirm | Esc=Quit", (10, ui_y_start + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(self.window_name, display_img)

    def run(self):
        while True:
            # Check for window close
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(20) & 0xFF
            
            if key == 27: # ESC
                break
            
            # Navigation
            # Left Arrow (usually 81 in Linux OpenCV GTK, 2420224 etc on Qt)
            # We check a few known codes for arrows mapping
            if key in [81, 2, 65361]: # Left variants
                new_val = max(0, self.current_frame_idx - 1)
                if new_val != self.current_frame_idx:
                   cv2.setTrackbarPos("Frame", self.window_name, new_val)
                   
            elif key in [83, 3, 65363]: # Right variants
                new_val = min(self.total_frames - 1, self.current_frame_idx + 1)
                if new_val != self.current_frame_idx:
                   cv2.setTrackbarPos("Frame", self.window_name, new_val)

            elif key == 13: # Enter
                coords = self.calculate_position()
                if coords is not None:
                    print(f"RESULT: Frame {self.current_frame_idx} -> X: {coords[0]:.4f} mm, Y: {coords[1]:.4f} mm")
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Video Annotation Tool")
    parser.add_argument("video_path", type=str, help="Relative path to video")
    parser.add_argument("length", type=float, help="Reference Rectangle Length (mm)")
    parser.add_argument("width", type=float, help="Reference Rectangle Width (mm)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: File '{args.video_path}' not found.")
        sys.exit(1)
        
    annotator = VideoAnnotator(args.video_path, args.length, args.width)
    annotator.run()

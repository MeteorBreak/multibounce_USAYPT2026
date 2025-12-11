import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter
import argparse

class BallTracker:
    def __init__(self):
        # 初始化ArUco字典和参数
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # A3纸尺寸 (单位:毫米)
        self.a3_width = 420
        self.a3_height = 297
        
        # 存储ArUco标记的角点位置
        self.marker_corners = {}
        
        # 球的追踪参数
        self.ball_positions = []
        self.min_radius = 30
        self.max_radius = 70
        self.param1 = 80  # 霍夫圆检测参数 - 边缘检测的高阈值
        self.param2 = 16  # 霍夫圆检测参数 - 圆心检测的累计阈值
        self.previous_ball_pos = None
        
        # 增强稳定性的参数
        self.position_history = deque(maxlen=10)  # 增加历史记录长度
        self.velocity_history = deque(maxlen=5)   # 速度历史记录
        self.confidence_scores = deque(maxlen=10) # 检测置信度
        self.lost_frames = 0                      # 连续未检测到球的帧数
        self.max_lost_frames = 5                  # 最大容忍连续丢失帧数
        self.max_speed = 50                       # 每帧最大移动距离 (像素)
        self.min_confidence = 0.5                 # 最小接受置信度
        
    def detect_aruco_markers(self, frame):
        """检测ArUco标记并返回标记信息和转换矩阵"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None:
            # 绘制检测到的标记
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 存储标记角点
            for i, marker_id in enumerate(ids):
                self.marker_corners[marker_id[0]] = corners[i][0]
        
        return corners, ids, frame
    
    def establish_coordinate_system(self, corners, ids):
        """根据ArUco标记建立坐标系"""
        if ids is None or len(ids) < 4:
            print("坐标系未建立，标记数量不足。")
            return None, None
        
        try:
            # 获取所有检测到的标记中心点
            detected_markers = {}
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in [0, 1, 2, 3]:
                    center = np.mean(corners[i][0], axis=0)
                    detected_markers[marker_id] = center

            if len(detected_markers) < 4:
                print(f"未检测到全部4个角点标记，当前检测到: {list(detected_markers.keys())}")
                return None, None

            # 对标记点进行排序，确保正确的对应关系
            # points_2d should be in order: top-left, top-right, bottom-right, bottom-left
            src_points = np.array([
                detected_markers[0], detected_markers[1],
                detected_markers[2], detected_markers[3]
            ], dtype=np.float32)

            # Sort points based on their y-coordinate
            y_sorted = src_points[np.argsort(src_points[:, 1]), :]
            
            # Sort the top two points by x-coordinate
            top_points = y_sorted[:2]
            top_points = top_points[np.argsort(top_points[:, 0]), :]
            
            # Sort the bottom two points by x-coordinate
            bottom_points = y_sorted[2:]
            bottom_points = bottom_points[np.argsort(bottom_points[:, 0]), :]

            # Final sorted 2D points
            points_2d = np.array([
                top_points[0],    # Top-left
                top_points[1],    # Top-right
                bottom_points[1], # Bottom-right
                bottom_points[0]  # Bottom-left
            ], dtype=np.float32)

            points_3d = np.array([
                [0, 0, 0],
                [self.a3_width, 0, 0],
                [self.a3_width, self.a3_height, 0],
                [0, self.a3_height, 0]
            ], dtype=np.float32)
            
            matrix, _ = cv2.findHomography(points_2d, points_3d[:, :2])
            print("坐标系建立成功。")
            return matrix, points_2d
        except Exception as e:
            print(f"坐标系建立异常：{e}")
            return None, None
    
    def detect_ball(self, frame):
        """检测视频帧中的球并增加置信度评分"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 100])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        gray_blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        ball_position = None
        confidence = 0.0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_circle, confidence = self._select_best_circle_with_confidence(circles[0], mask)
            if best_circle is not None and confidence >= self.min_confidence:
                x, y, r = best_circle
                ball_position = (x, y)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                self.previous_ball_pos = ball_position
                self.lost_frames = 0
            else:
                self.lost_frames += 1
        else:
            self.lost_frames += 1
        if ball_position is None and self.lost_frames < self.max_lost_frames and len(self.position_history) >= 2:
            predicted_pos = self._predict_position()
            if predicted_pos is not None:
                cv2.circle(frame, predicted_pos, 5, (0, 165, 255), -1)
                cv2.putText(frame, "Predicted", (predicted_pos[0]+10, predicted_pos[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        return frame, ball_position, confidence
    
    def _select_best_circle_with_confidence(self, circles, mask):
        if len(circles) == 0:
            return None, 0.0
        if len(circles) == 1:
            x, y, r = circles[0]
            confidence = self._calculate_circle_confidence(x, y, r, mask)
            return circles[0], confidence
        best_circle = None
        best_score = -1
        best_confidence = 0.0
        for circle in circles:
            x, y, r = circle
            confidence = self._calculate_circle_confidence(x, y, r, mask)
            score = confidence
            if self.previous_ball_pos is not None:
                distance = np.sqrt((x - self.previous_ball_pos[0])**2 + (y - self.previous_ball_pos[1])**2)
                if distance <= self.max_speed:
                    distance_score = max(0, 1 - distance / self.max_speed)
                    score *= (1 + distance_score)
                else:
                    score *= 0.5
            if score > best_score:
                best_score = score
                best_circle = circle
                best_confidence = confidence
        return best_circle, best_confidence
    
    def _calculate_circle_confidence(self, x, y, r, mask):
        center_region = mask[max(0, y-r//2):min(mask.shape[0], y+r//2), 
                            max(0, x-r//2):min(mask.shape[1], x+r//2)]
        if center_region.size == 0:
            return 0
        white_density = np.sum(center_region > 0) / center_region.size
        radius_score = 1.0
        if r < self.min_radius:
            radius_score = max(0.3, r / self.min_radius)
        elif r > self.max_radius:
            radius_score = max(0.3, self.max_radius / r)
        edge_distance = min(x, y, mask.shape[1]-x, mask.shape[0]-y)
        edge_factor = min(1.0, edge_distance / (r*2))
        confidence = white_density * radius_score * edge_factor
        return confidence
    
    def _predict_position(self):
        if len(self.position_history) < 2:
            return None
        recent_positions = list(self.position_history)
        if len(recent_positions) >= 2:
            last_pos = recent_positions[-1]
            prev_pos = recent_positions[-2]
            dx = last_pos[0] - prev_pos[0]
            dy = last_pos[1] - prev_pos[1]
            predicted_x = int(last_pos[0] + dx)
            predicted_y = int(last_pos[1] + dy)
            return (predicted_x, predicted_y)
        return None
    
    def _smooth_position(self, current_position, confidence):
        if current_position is None:
            return None
        self.position_history.append(current_position)
        self.confidence_scores.append(confidence)
        if len(self.position_history) < 3:
            return current_position
        weights = np.array(list(self.confidence_scores))
        time_weights = np.linspace(0.5, 1.0, len(weights))
        weights = weights * time_weights
        weights = weights / np.sum(weights)
        if len(self.position_history) >= 2:
            last_pos = self.position_history[-2]
            current_pos = self.position_history[-1]
            distance = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            if distance > self.max_speed:
                weights[-1] = weights[-1] * 0.5
                weights = weights / np.sum(weights)
        positions = np.array(list(self.position_history))
        smoothed_x = np.sum(positions[:, 0] * weights)
        smoothed_y = np.sum(positions[:, 1] * weights)
        if len(self.position_history) >= 5:
            try:
                x_coords = positions[:, 0]
                y_coords = positions[:, 1]
                window_length = min(5, len(x_coords) - (len(x_coords) % 2 == 0))
                if window_length >= 3:
                    x_smooth = savgol_filter(x_coords, window_length, 2)
                    y_smooth = savgol_filter(y_coords, window_length, 2)
                    smoothed_x = x_smooth[-1]
                    smoothed_y = y_smooth[-1]
            except:
                pass
        return (int(smoothed_x), int(smoothed_y))
    
    def _calculate_velocity(self):
        if len(self.position_history) < 2:
            return None
        recent_positions = list(self.position_history)
        last_pos = recent_positions[-1]
        prev_pos = recent_positions[-2]
        dx = last_pos[0] - prev_pos[0]
        dy = last_pos[1] - prev_pos[1]
        self.velocity_history.append((dx, dy))
        return (dx, dy)
    
    def _is_valid_position(self, real_world_coord):
        if real_world_coord is None:
            print("真实坐标为None，判为无效。")
            return False
        x, y = real_world_coord
        margin = 50
        if not (-margin <= x <= self.a3_width + margin and -margin <= y <= self.a3_height + margin):
            print(f"坐标越界：({x}, {y})")
            return False
        if len(self.ball_positions) > 0:
            last_pos = self.ball_positions[-1]
            distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
            max_distance_per_frame = 100
            if distance > max_distance_per_frame:
                if len(self.ball_positions) > 1:
                    prev_pos = self.ball_positions[-2]
                    prev_dx = last_pos[0] - prev_pos[0]
                    prev_dy = last_pos[1] - prev_pos[1]
                    current_dx = x - last_pos[0]
                    current_dy = y - last_pos[1]
                    prev_mag = np.sqrt(prev_dx**2 + prev_dy**2)
                    current_mag = np.sqrt(current_dx**2 + current_dy**2)
                    if prev_mag > 0 and current_mag > 0:
                        direction_cosine = (prev_dx*current_dx + prev_dy*current_dy) / (prev_mag*current_mag)
                        if direction_cosine < 0:
                            print(f"运动方向异常，余弦={direction_cosine}")
                            return False
                if distance > 2 * max_distance_per_frame:
                    print(f"单帧位移过大: {distance}")
                    return False
        return True
    
    def process_video(self, video_path=None, frames_to_process=None):
        if video_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        homography_matrix = None
        marker_points = None
        
        # 如果指定了要处理的帧
        if frames_to_process:
            for frame_num in frames_to_process:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if not ret:
                    print(f"无法读取第 {frame_num} 帧。")
                    continue

                corners, ids, frame = self.detect_aruco_markers(frame)
                if homography_matrix is None:
                    homography_matrix, marker_points = self.establish_coordinate_system(corners, ids)
                
                frame, ball_position, confidence = self.detect_ball(frame)

                if ball_position is not None and homography_matrix is not None:
                    smoothed_position = self._smooth_position(ball_position, confidence)
                    if smoothed_position:
                        self._calculate_velocity()
                        real_world_coord = self.transform_to_real_world(smoothed_position, homography_matrix)
                        if self._is_valid_position(real_world_coord):
                            self.ball_positions.append(real_world_coord)
                            cv2.putText(frame, f"Position: ({real_world_coord[0]:.1f}, {real_world_coord[1]:.1f}) mm", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Position filtered (outlier)", 
                                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                info_text = f"Frame: {frame_num}, Param1: {self.param1}, Param2: {self.param2}, Radius: {self.min_radius}-{self.max_radius}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                detection_count = f"Detections: {len(self.ball_positions)}"
                cv2.putText(frame, detection_count, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 将帧编号绘制在图像上
                frame_text = f"Frame: {frame_num}"
                text_size, _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = frame.shape[1] - text_size[0] - 20
                text_y = 50
                cv2.putText(frame, frame_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imshow(f'Ball Tracking - Frame {frame_num}', frame)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()

            cap.release()
            print(f"最终记录球坐标数量: {len(self.ball_positions)}")
            return self.ball_positions

        # 如果未指定帧，则按原方式处理视频流
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频读取结束或失败。")
                break
            frame_count += 1

            corners, ids, frame = self.detect_aruco_markers(frame)
            if homography_matrix is None:
                homography_matrix, marker_points = self.establish_coordinate_system(corners, ids)
                if homography_matrix is None:
                    print(f"第{frame_count}帧未建立坐标系，无法转换坐标。")
            frame, ball_position, confidence = self.detect_ball(frame)
            if ball_position is not None:
                print(f"第{frame_count}帧检测到球像素位置: {ball_position}, 置信度: {confidence:.2f}")
            else:
                print(f"第{frame_count}帧未检测到球。")
            if ball_position is not None and homography_matrix is not None:
                smoothed_position = self._smooth_position(ball_position, confidence)
                print(f"第{frame_count}帧平滑后像素位置: {smoothed_position}")
                if smoothed_position:
                    self._calculate_velocity()
                    real_world_coord = self.transform_to_real_world(smoothed_position, homography_matrix)
                    print(f"第{frame_count}帧转换到真实坐标: {real_world_coord}")
                    if self._is_valid_position(real_world_coord):
                        self.ball_positions.append(real_world_coord)
                        print(f"第{frame_count}帧记录到有效真实坐标: {real_world_coord}")
                        cv2.putText(frame, f"Position: ({real_world_coord[0]:.1f}, {real_world_coord[1]:.1f}) mm", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        print(f"第{frame_count}帧坐标被判为异常值: {real_world_coord}")
                        cv2.putText(frame, "Position filtered (outlier)", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            info_text = f"Param1: {self.param1}, Param2: {self.param2}, Radius: {self.min_radius}-{self.max_radius}"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            detection_count = f"Detections: {len(self.ball_positions)}"
            cv2.putText(frame, detection_count, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Ball Tracking', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                self.adjust_detection_parameters(increase_sensitivity=True)
                print("Increased sensitivity:", self.get_detection_info())
            elif key == ord('-'):
                self.adjust_detection_parameters(increase_sensitivity=False)
                print("Decreased sensitivity:", self.get_detection_info())
            elif key == ord('r'):
                self.param1 = 80
                self.param2 = 25
                self.min_radius = 35
                self.max_radius = 60
                print("Reset parameters:", self.get_detection_info())
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"最终记录球坐标数量: {len(self.ball_positions)}")
        return self.ball_positions

    def transform_to_real_world(self, point, homography_matrix):
        if homography_matrix is None:
            print("转换矩阵为None，无法转换坐标。")
            return None
        point_homogeneous = np.array([point[0], point[1], 1])
        transformed = np.dot(homography_matrix, point_homogeneous)
        transformed = transformed / transformed[2]
        return (transformed[0], transformed[1])
        
    def adjust_detection_parameters(self, increase_sensitivity=True):
        if increase_sensitivity:
            self.param1 = max(30, self.param1 - 10)
            self.param2 = max(15, self.param2 - 5)
            self.min_radius = max(20, self.min_radius - 5)
            self.max_radius = min(80, self.max_radius + 5)
        else:
            self.param1 = min(100, self.param1 + 10)
            self.param2 = min(50, self.param2 + 5)
            self.min_radius = min(40, self.min_radius + 5)
            self.max_radius = max(50, self.max_radius - 5)
    
    def get_detection_info(self):
        return {
            'param1': self.param1,
            'param2': self.param2,
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
            'positions_count': len(self.ball_positions)
        }
        
    def plot_results(self):
        if not self.ball_positions:
            print("No ball positions recorded.")
            return
        x_coords = [pos[0] for pos in self.ball_positions]
        y_coords = [pos[1] for pos in self.ball_positions]
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c='red', marker='o', alpha=0.7)
        plt.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=1)
        plt.xlabel('X Coordinate (mm)')
        plt.ylabel('Y Coordinate (mm)')
        plt.title('Ball Landing Positions')
        plt.grid(True)
        plt.axis('equal')
        plt.axhline(y=0, color='blue', linestyle='--')
        plt.axhline(y=self.a3_height, color='blue', linestyle='--')
        plt.axvline(x=0, color='blue', linestyle='--')
        plt.axvline(x=self.a3_width, color='blue', linestyle='--')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track a ball in a video.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')
    parser.add_argument('--frames', nargs='*', type=int, help='Optional list of frame numbers to process.')
    args = parser.parse_args()

    tracker = BallTracker()
    positions = tracker.process_video(args.video_path, frames_to_process=args.frames)
    print("Recorded ball positions (mm):")
    for i, pos in enumerate(positions):
        print(f"Position {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")
    tracker.plot_results()

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt

class BallTracker:
    def __init__(self):
        # 初始化ArUco字典和参数
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # A3纸尺寸 (单位:毫米)
        self.a3_width = 297
        self.a3_height = 420
        
        # 存储ArUco标记的角点位置
        self.marker_corners = {}
        
        # 球的追踪参数
        self.ball_positions = []
        self.min_radius = 35
        self.max_radius = 60
        self.param1 = 80  # 霍夫圆检测参数 - 边缘检测的高阈值
        self.param2 = 25  # 霍夫圆检测参数 - 圆心检测的累计阈值
        self.previous_ball_pos = None
        self.detection_confidence = []
        self.position_history = []  # 存储最近几帧的位置
        self.history_size = 5  # 保留的历史帧数
        
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
            return None, None
        
        # 这里假设四个标记的ID为0,1,2,3，分别对应A3纸的四个角
        # 实际使用时需要根据你的标记ID进行调整
        try:
            # 获取四个角点的中心坐标
            points_2d = []
            for i in range(4):
                if i not in self.marker_corners:
                    return None, None
                marker_corner = self.marker_corners[i]
                center = np.mean(marker_corner, axis=0)
                points_2d.append(center)
            
            points_2d = np.array(points_2d, dtype=np.float32)
            
            # 定义A3纸的实际物理坐标 (单位:毫米)
            points_3d = np.array([
                [0, 0, 0],               # 左下角
                [self.a3_width, 0, 0],   # 右下角
                [self.a3_width, self.a3_height, 0],  # 右上角
                [0, self.a3_height, 0]   # 左上角
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            matrix, _ = cv2.findHomography(points_2d, points_3d[:, :2])
            
            return matrix, points_2d
        except:
            return None, None
    
    def detect_ball(self, frame):
        """检测视频帧中的球"""
        # 转换为HSV色彩空间以便更好地分离白色球体
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定义白色的HSV范围
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # 创建白色掩膜
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用掩膜到灰度图
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        # 高斯模糊减少噪声
        gray_blurred = cv2.GaussianBlur(gray_masked, (9, 9), 2)
        
        # 使用霍夫圆变换检测球
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
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # 如果检测到多个圆，进行筛选
            best_circle = self._select_best_circle(circles[0], mask)
            
            if best_circle is not None:
                x, y, r = best_circle
                ball_position = (x, y)
                
                # 绘制球
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
                
                # 更新前一帧位置
                self.previous_ball_pos = ball_position
        
        return frame, ball_position
    
    def _select_best_circle(self, circles, mask):
        """从多个检测到的圆中选择最佳的球"""
        if len(circles) == 0:
            return None
        
        # 如果只有一个圆，直接返回
        if len(circles) == 1:
            return circles[0]
        
        best_circle = None
        best_score = -1
        
        for circle in circles:
            x, y, r = circle
            
            # 计算圆的质量评分
            score = self._calculate_circle_score(x, y, r, mask)
            
            # 如果有前一帧的位置，考虑距离因素
            if self.previous_ball_pos is not None:
                distance = np.sqrt((x - self.previous_ball_pos[0])**2 + (y - self.previous_ball_pos[1])**2)
                # 距离越近，得分越高（假设球移动不会太快）
                distance_score = max(0, 1 - distance / 100.0)
                score *= (1 + distance_score)
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        return best_circle
    
    def _calculate_circle_score(self, x, y, r, mask):
        """计算圆的质量评分"""
        # 检查圆心周围的像素密度
        center_region = mask[max(0, y-r//2):min(mask.shape[0], y+r//2), 
                            max(0, x-r//2):min(mask.shape[1], x+r//2)]
        
        if center_region.size == 0:
            return 0
        
        # 白色像素密度
        white_density = np.sum(center_region > 0) / center_region.size
        
        # 圆形度评分（基于半径的合理性）
        radius_score = 1.0
        if r < self.min_radius or r > self.max_radius:
            radius_score = 0.5
        
        return white_density * radius_score
    
    def _smooth_position(self, current_position):
        """使用历史位置对当前位置进行平滑处理"""
        if current_position is None:
            return None
        
        # 添加当前位置到历史记录
        self.position_history.append(current_position)
        
        # 保持历史记录大小
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
        
        # 如果历史记录不足，直接返回当前位置
        if len(self.position_history) < 3:
            return current_position
        
        # 使用加权平均进行平滑
        weights = np.array([0.1, 0.2, 0.3, 0.4])[:len(self.position_history)]
        weights = weights / np.sum(weights)
        
        smoothed_x = sum(pos[0] * w for pos, w in zip(self.position_history, weights))
        smoothed_y = sum(pos[1] * w for pos, w in zip(self.position_history, weights))
        
        return (int(smoothed_x), int(smoothed_y))
    
    def _is_valid_position(self, real_world_coord):
        """检查位置是否为有效值（异常值检测）"""
        if real_world_coord is None:
            return False
        
        x, y = real_world_coord
        
        # 检查是否在A3纸范围内（允许一定误差）
        margin = 50  # 50mm的误差边界
        if not (-margin <= x <= self.a3_width + margin and -margin <= y <= self.a3_height + margin):
            return False
        
        # 如果有历史位置，检查移动距离是否合理
        if len(self.ball_positions) > 0:
            last_pos = self.ball_positions[-1]
            distance = np.sqrt((x - last_pos[0])**2 + (y - last_pos[1])**2)
            # 假设球在一帧间不会移动超过100mm
            if distance > 100:
                return False
        
        return True
    
    def adjust_detection_parameters(self, increase_sensitivity=True):
        """动态调整检测参数以改善检测效果"""
        if increase_sensitivity:
            # 提高敏感度
            self.param1 = max(30, self.param1 - 10)
            self.param2 = max(15, self.param2 - 5)
            self.min_radius = max(20, self.min_radius - 5)
            self.max_radius = min(80, self.max_radius + 5)
        else:
            # 降低敏感度
            self.param1 = min(100, self.param1 + 10)
            self.param2 = min(50, self.param2 + 5)
            self.min_radius = min(40, self.min_radius + 5)
            self.max_radius = max(50, self.max_radius - 5)
    
    def get_detection_info(self):
        """获取当前检测参数信息"""
        return {
            'param1': self.param1,
            'param2': self.param2,
            'min_radius': self.min_radius,
            'max_radius': self.max_radius,
            'positions_count': len(self.ball_positions)
        }
    
    def transform_to_real_world(self, point, homography_matrix):
        """将像素坐标转换为真实世界坐标"""
        if homography_matrix is None:
            return None
        
        # 转换为齐次坐标
        point_homogeneous = np.array([point[0], point[1], 1])
        
        # 应用变换矩阵
        transformed = np.dot(homography_matrix, point_homogeneous)
        
        # 转换回笛卡尔坐标
        transformed = transformed / transformed[2]
        
        return (transformed[0], transformed[1])
    
    def process_video(self, video_path=None):
        """处理视频并追踪球的位置"""
        if video_path is None:
            cap = cv2.VideoCapture(0)  # 使用默认摄像头
        else:
            cap = cv2.VideoCapture(video_path)
        
        homography_matrix = None
        marker_points = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测ArUco标记
            corners, ids, frame = self.detect_aruco_markers(frame)
            
            # 建立坐标系（如果还没有建立）
            if homography_matrix is None:
                homography_matrix, marker_points = self.establish_coordinate_system(corners, ids)
            
            # 检测球
            frame, ball_position = self.detect_ball(frame)
            
            # 如果检测到球并且有转换矩阵，则计算真实坐标
            if ball_position is not None and homography_matrix is not None:
                # 应用位置平滑
                smoothed_position = self._smooth_position(ball_position)
                
                real_world_coord = self.transform_to_real_world(smoothed_position, homography_matrix)
                
                # 异常值检测
                if self._is_valid_position(real_world_coord):
                    self.ball_positions.append(real_world_coord)
                    
                    # 在图像上显示坐标
                    cv2.putText(frame, f"Position: ({real_world_coord[0]:.1f}, {real_world_coord[1]:.1f}) mm", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # 显示异常检测信息
                    cv2.putText(frame, "Position filtered (outlier)", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示检测参数信息
            info_text = f"Param1: {self.param1}, Param2: {self.param2}, Radius: {self.min_radius}-{self.max_radius}"
            cv2.putText(frame, info_text, (10, frame.shape[0] - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            detection_count = f"Detections: {len(self.ball_positions)}"
            cv2.putText(frame, detection_count, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示结果
            cv2.imshow('Ball Tracking', frame)
            
            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'): # 增加敏感度
                self.adjust_detection_parameters(increase_sensitivity=True)
                print("Increased sensitivity:", self.get_detection_info())
            elif key == ord('-'): # 降低敏感度
                self.adjust_detection_parameters(increase_sensitivity=False)
                print("Decreased sensitivity:", self.get_detection_info())
            elif key == ord('r'): # 重置参数
                self.param1 = 80
                self.param2 = 25
                self.min_radius = 35
                self.max_radius = 60
                print("Reset parameters:", self.get_detection_info())
        
        cap.release()
        cv2.destroyAllWindows()
        
        return self.ball_positions

    def plot_results(self):
        """绘制球的位置结果"""
        if not self.ball_positions:
            print("No ball positions recorded.")
            return
        
        # 提取x和y坐标
        x_coords = [pos[0] for pos in self.ball_positions]
        y_coords = [pos[1] for pos in self.ball_positions]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(x_coords, y_coords, c='red', marker='o')
        plt.xlabel('X Coordinate (mm)')
        plt.ylabel('Y Coordinate (mm)')
        plt.title('Ball Landing Positions')
        plt.grid(True)
        
        # 设置坐标轴比例相等
        plt.axis('equal')
        
        # 显示A3纸边界
        plt.axhline(y=0, color='blue', linestyle='--')
        plt.axhline(y=self.a3_height, color='blue', linestyle='--')
        plt.axvline(x=0, color='blue', linestyle='--')
        plt.axvline(x=self.a3_width, color='blue', linestyle='--')
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    tracker = BallTracker()
    
    # 处理视频文件（如果提供视频文件路径）
    positions = tracker.process_video("videos/test7.mp4")
    
    # 或者使用摄像头实时处理
    # positions = tracker.process_video()
    
    # 打印所有记录的位置
    print("Recorded ball positions (mm):")
    for i, pos in enumerate(positions):
        print(f"Position {i+1}: ({pos[0]:.2f}, {pos[1]:.2f})")
    
    # 绘制结果
    tracker.plot_results()
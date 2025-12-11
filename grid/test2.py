import cv2
import numpy as np

class PingPongBallDetector:
    def __init__(self):
        # 乒乓球物理参数（单位：毫米）
        self.ball_diameter_mm = 40  # 标准乒乓球直径约40mm
        self.min_ball_area = 100    # 最小面积阈值（像素）
        self.max_ball_area = 2000   # 最大面积阈值（像素）
        
        # 检测参数
        self.blur_kernel_size = 5   # 高斯模糊核大小
        self.threshold_value = 200  # 二值化阈值（0-255）
        self.min_circularity = 0.7  # 最小圆形度
        
        # 跟踪参数
        self.max_movement_per_frame = 50  # 最大帧间移动距离（像素）
        self.tracked_ball = None          # 跟踪的球位置
        self.tracking_frames = 0          # 连续跟踪帧数
        
    def preprocess_image(self, image):
        """图像预处理"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # 直方图均衡化增强对比度（可选，根据光照条件决定是否启用）
        # equalized = cv2.equalizeHist(blurred)
        
        return blurred
    
    def adaptive_thresholding(self, image):
        """自适应阈值处理，应对光照不均"""
        # 使用自适应阈值代替全局阈值
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # 邻域大小
            2    # 常数减去的值
        )
        return binary
    
    def detect_balls(self, image):
        """检测乒乓球"""
        # 预处理
        processed = self.preprocess_image(image)
        
        # 二值化 - 尝试两种方法，选择效果更好的
        # 方法1: 自适应阈值（应对光照不均）
        binary_adaptive = self.adaptive_thresholding(processed)
        
        # 方法2: 全局阈值（简单场景效果更好）
        _, binary_global = cv2.threshold(processed, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # 形态学操作 - 先开后闭
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
        binary_adaptive = cv2.morphologyEx(binary_adaptive, cv2.MORPH_CLOSE, kernel)
        
        binary_global = cv2.morphologyEx(binary_global, cv2.MORPH_OPEN, kernel)
        binary_global = cv2.morphologyEx(binary_global, cv2.MORPH_CLOSE, kernel)
        
        # 检测轮廓
        contours_adaptive, _ = cv2.findContours(binary_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_global, _ = cv2.findContours(binary_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 合并轮廓
        all_contours = contours_adaptive + contours_global
        
        # 过滤轮廓
        candidates = []
        for contour in all_contours:
            # 面积过滤
            area = cv2.contourArea(contour)
            if area < self.min_ball_area or area > self.max_ball_area:
                continue
                
            # 圆形度过滤
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue
                
            # 凸性检测（球应该是凸的）
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
                
            solidity = area / hull_area
            if solidity < 0.8:  # 凸性阈值
                continue
                
            # 边界框纵横比
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # 球应该接近圆形
                continue
                
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            candidates.append({
                "contour": contour,
                "center": (cx, cy),
                "area": area,
                "circularity": circularity
            })
        
        return candidates, binary_adaptive, binary_global
    
    def track_ball(self, candidates, frame_shape):
        """基于运动连续性跟踪球"""
        best_candidate = None
        
        # 如果没有跟踪记录，选择最符合条件的候选
        if self.tracked_ball is None:
            if candidates:
                # 按圆形度排序，选择最圆的
                candidates.sort(key=lambda x: x["circularity"], reverse=True)
                best_candidate = candidates[0]
                self.tracked_ball = best_candidate["center"]
                self.tracking_frames = 1
        else:
            # 有跟踪记录，选择距离上次位置最近的候选
            min_distance = float('inf')
            for candidate in candidates:
                cx, cy = candidate["center"]
                tx, ty = self.tracked_ball
                distance = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                
                if distance < min_distance and distance < self.max_movement_per_frame:
                    min_distance = distance
                    best_candidate = candidate
            
            if best_candidate:
                self.tracked_ball = best_candidate["center"]
                self.tracking_frames += 1
            else:
                # 丢失跟踪
                self.tracking_frames = max(0, self.tracking_frames - 1)
                if self.tracking_frames == 0:
                    self.tracked_ball = None
        
        return best_candidate
    
    def detect_contact_frame(self, ball_positions, min_contact_frames=3):
        """检测球触地的帧"""
        if len(ball_positions) < min_contact_frames:
            return None
            
        # 计算球在垂直方向的速度
        y_positions = [pos[1] for pos in ball_positions]
        
        # 寻找最低点（触地点）
        min_y_index = np.argmax(y_positions)  # y坐标越大，位置越低
        min_y_frame = len(y_positions) - min_y_index - 1  # 转换为时间顺序
        
        # 验证是否确实是触地（速度变化）
        if min_y_index > 0 and min_y_index < len(y_positions) - 1:
            # 计算触地前后的速度
            before_contact = y_positions[min_y_index] - y_positions[min_y_index - 1]
            after_contact = y_positions[min_y_index + 1] - y_positions[min_y_index]
            
            # 触地前向下运动（速度为正），触地后向上运动（速度为负）
            if before_contact > 0 and after_contact < 0:
                return min_y_frame
        
        return None

# 使用示例
def process_video(video_path, output_path):
    """处理视频并检测乒乓球落点"""
    cap = cv2.VideoCapture(video_path)
    detector = PingPongBallDetector()
    
    # 存储球的位置用于触地检测
    ball_positions = []
    contact_frame = None
    
    # 视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                         (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 检测乒乓球
        candidates, binary_adaptive, binary_global = detector.detect_balls(frame)
        ball = detector.track_ball(candidates, frame.shape)
        
        # 绘制结果
        result_frame = frame.copy()
        
        # 如果检测到球
        if ball:
            # 绘制球轮廓和中心
            cv2.drawContours(result_frame, [ball["contour"]], -1, (0, 255, 0), 2)
            cv2.circle(result_frame, ball["center"], 5, (0, 0, 255), -1)
            
            # 记录球位置
            ball_positions.append(ball["center"])
            
            # 显示球信息
            cv2.putText(result_frame, f"Ball: {ball['center']}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 检查是否触地
        if contact_frame is None and len(ball_positions) > 10:
            contact_frame = detector.detect_contact_frame(ball_positions[-10:])
            if contact_frame is not None:
                actual_frame = frame_count - (10 - contact_frame)
                cv2.putText(result_frame, f"CONTACT at frame {actual_frame}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Ball contacted at frame {actual_frame}")
        
        # 显示帧号
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 写入输出视频
        out.write(result_frame)
        
        # 显示结果（可选）
        cv2.imshow('Ping Pong Ball Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return ball_positions, contact_frame

# 如果直接运行此脚本，处理示例视频
if __name__ == "__main__":
    # 替换为您的视频路径
    video_path = "videos/test7.mp4"
    output_path = "output_video.avi"
    
    ball_positions, contact_frame = process_video(video_path, output_path)
    
    if contact_frame is not None:
        print(f"Ball contacted the surface at frame {contact_frame}")
        if ball_positions:
            contact_position = ball_positions[contact_frame]
            print(f"Contact position: {contact_position}")
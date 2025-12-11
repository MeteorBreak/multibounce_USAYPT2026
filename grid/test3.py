import cv2
import numpy as np

class SimpleBallDetector:
    def __init__(self):
        # 基本参数
        self.min_radius = 10   # 最小半径(像素)
        self.max_radius = 50   # 最大半径(像素)
        self.threshold = 200   # 二值化阈值
        
        # 跟踪参数
        self.last_position = None
        self.tracking_confirmed = False
        
    def detect_ball(self, frame):
        """检测乒乓球位置"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 二值化 - 白色乒乓球在黑色背景上
        _, binary = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 寻找最可能是球的轮廓
        ball_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 面积过滤
            if area < 100 or area > 3000:
                continue
                
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # 半径过滤
            if radius < self.min_radius or radius > self.max_radius:
                continue
                
            # 圆形度检查
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area if circle_area > 0 else 0
            
            if circularity < 0.6:  # 不够圆
                continue
                
            # 选择最大的合适轮廓
            if area > max_area:
                max_area = area
                ball_contour = contour
                
        return ball_contour
    
    def track_ball(self, frame):
        """跟踪球的位置"""
        ball_contour = self.detect_ball(frame)
        
        if ball_contour is None:
            # 如果没有检测到球，但之前有跟踪记录，尝试在附近搜索
            if self.last_position and self.tracking_confirmed:
                x, y = self.last_position
                # 在上一帧位置附近创建ROI
                h, w = frame.shape[:2]
                margin = 50
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(w, x + margin)
                y2 = min(h, y + margin)
                
                roi = frame[y1:y2, x1:x2]
                roi_contour = self.detect_ball(roi)
                
                if roi_contour is not None:
                    # 计算在原始图像中的位置
                    M = cv2.moments(roi_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + x1
                        cy = int(M["m01"] / M["m00"]) + y1
                        self.last_position = (cx, cy)
                        return (cx, cy), roi_contour
                
            self.tracking_confirmed = False
            return None, None
        
        # 计算轮廓中心
        M = cv2.moments(ball_contour)
        if M["m00"] == 0:
            return None, None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # 简单的运动连续性检查
        if self.last_position:
            prev_x, prev_y = self.last_position
            distance = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
            
            # 如果移动距离合理，确认跟踪
            if distance < 100:  # 最大合理移动距离
                self.tracking_confirmed = True
            else:
                # 移动距离过大，可能是误检
                if not self.tracking_confirmed:
                    return None, None
        
        self.last_position = (cx, cy)
        return (cx, cy), ball_contour

def detect_ball_simple(frame, threshold=200):
    """最简单的单帧检测函数"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 二值化
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 寻找最大的合适轮廓
    if contours:
        # 按面积排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 面积太小
                continue
                
            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            if radius < 10 or radius > 50:  # 半径不合理
                continue
                
            # 计算中心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            return (cx, cy), contour
            
    return None, None

# 使用示例
def process_video_simple(video_path):
    """处理视频的简单示例"""
    cap = cv2.VideoCapture(video_path)
    detector = SimpleBallDetector()
    
    positions = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 方法1: 使用跟踪器
        position, contour = detector.track_ball(frame)
        
        # 方法2: 或者直接使用单帧检测
        # position, contour = detect_ball_simple(frame)
        
        if position:
            positions.append((frame_count, position))
            
            # 在图像上绘制结果
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, position, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Ball: {position}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示帧
        cv2.imshow('Ball Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    return positions

# 如果只有单张图片
def detect_ball_in_image(image_path):
    """在单张图片中检测球"""
    image = cv2.imread(image_path)
    position, contour = detect_ball_simple(image)
    
    if position:
        # 绘制结果
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.circle(image, position, 5, (0, 0, 255), -1)
        
        # 显示结果
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return position
    
    return None

# 调整阈值以优化检测
def find_optimal_threshold(image_path):
    """交互式调整阈值找到最佳值"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def update_threshold(val):
        threshold = val
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制所有轮廓
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        cv2.imshow('Threshold Tuning', result)
    
    cv2.namedWindow('Threshold Tuning')
    cv2.createTrackbar('Threshold', 'Threshold Tuning', 200, 255, update_threshold)
    
    # 初始显示
    update_threshold(200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    positions = process_video_simple("videos/test1.mp4")
    
    # 处理单张图片
    # position = detect_ball_in_image("ball_image.jpg")
    
    # 调整阈值
    # find_optimal_threshold("ball_image.jpg")
    pass
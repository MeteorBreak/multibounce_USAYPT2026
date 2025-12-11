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
        self.min_radius = 40
        self.max_radius = 50
        self.param1 = 50  # 霍夫圆检测参数
        self.param2 = 30  # 霍夫圆检测参数
        
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
    
    def preprocess_frame(self, frame):
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # 自适应阈值化
        _, thresholded = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        return thresholded

    def detect_ball(self, frame):
        """检测视频帧中的球"""
        # 预处理图像
        preprocessed = self.preprocess_frame(frame)
        
        # 使用霍夫圆变换检测球
        circles = cv2.HoughCircles(
            preprocessed, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=50,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        ball_position = None
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 选择最大的圆作为球
            largest_circle = max(circles[0, :], key=lambda x: x[2])
            
            # 提取球的中心坐标和半径
            x, y, r = largest_circle
            ball_position = (x, y)
            
            # 验证圆的半径是否合理
            if self.min_radius <= r <= self.max_radius:
                # 绘制球
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)
        
        return frame, ball_position



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
                real_world_coord = self.transform_to_real_world(ball_position, homography_matrix)
                self.ball_positions.append(real_world_coord)
                
                # 在图像上显示坐标
                cv2.putText(frame, f"Position: ({real_world_coord[0]:.1f}, {real_world_coord[1]:.1f}) mm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Ball Tracking', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
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
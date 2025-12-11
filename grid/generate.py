import cv2
import numpy as np
import matplotlib.pyplot as plt

# A3尺寸 (单位:毫米)
A3_WIDTH_MM = 420
A3_HEIGHT_MM = 297

# DPI设置 (打印质量)
DPI = 300

# 转换毫米到像素
def mm_to_pixels(mm, dpi):
    return int(mm * dpi / 25.4)

# 创建A3画布
width_px = mm_to_pixels(A3_WIDTH_MM, DPI)
height_px = mm_to_pixels(A3_HEIGHT_MM, DPI)
canvas = np.zeros((height_px, width_px, 3), dtype=np.uint8)  # 黑色背景

# 网格参数
GRID_SPACING_MM = 10  # 网格间距10mm
GRID_SPACING_PX = mm_to_pixels(GRID_SPACING_MM, DPI)
MAJOR_GRID_SPACING_MM = 50  # 主网格间距50mm
MAJOR_GRID_SPACING_PX = mm_to_pixels(MAJOR_GRID_SPACING_MM, DPI)

'''
# 绘制网格线
for x in range(0, width_px, GRID_SPACING_PX):
    color = (200, 200, 200)  # 浅灰色
    thickness = 1
    # 每5条线绘制一条粗线
    if x % MAJOR_GRID_SPACING_PX == 0:
        color = (150, 150, 150)  # 深灰色
        thickness = 2
    cv2.line(canvas, (x, 0), (x, height_px), color, thickness)

for y in range(0, height_px, GRID_SPACING_PX):
    color = (200, 200, 200)  # 浅灰色
    thickness = 1
    # 每5条线绘制一条粗线
    if y % MAJOR_GRID_SPACING_PX == 0:
        color = (150, 150, 150)  # 深灰色
        thickness = 2
    cv2.line(canvas, (0, y), (width_px, y), color, thickness)
'''

# 创建ArUco标记
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size_px = mm_to_pixels(50, DPI)  # 标记尺寸50mm

# 在四个角放置标记
marker_ids = [0, 1, 2, 3]
margin = mm_to_pixels(20, DPI)  # 边距20mm

# 左上角
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[0], marker_size_px)
# 将单通道标记转换为三通道
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
x_start, y_start = margin, margin
canvas[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img_bgr

# 右上角
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[1], marker_size_px)
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
x_start = width_px - margin - marker_size_px
y_start = margin
canvas[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img_bgr

# 左下角
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[2], marker_size_px)
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
x_start = margin
y_start = height_px - margin - marker_size_px
canvas[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img_bgr

# 右下角
marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_ids[3], marker_size_px)
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
x_start = width_px - margin - marker_size_px
y_start = height_px - margin - marker_size_px
canvas[y_start:y_start+marker_size_px, x_start:x_start+marker_size_px] = marker_img_bgr

# 添加坐标文本（可选）
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (100, 100, 100)
font_thickness = 1

# 在主要网格线上添加坐标标签
for x in range(0, width_px, MAJOR_GRID_SPACING_PX):
    if x > margin and x < width_px - margin:  # 避免与ArUco标记重叠
        coord_mm = int(x * 25.4 / DPI)
        cv2.putText(canvas, f"{coord_mm}", (x, margin//2), font, 
                   font_scale, font_color, font_thickness, cv2.LINE_AA)

for y in range(0, height_px, MAJOR_GRID_SPACING_PX):
    if y > margin and y < height_px - margin:  # 避免与ArUco标记重叠
        coord_mm = int(y * 25.4 / DPI)
        cv2.putText(canvas, f"{coord_mm}", (margin//2, y), font, 
                   font_scale, font_color, font_thickness, cv2.LINE_AA)

# 保存图像
output_path = "A3_grid_with_aruco_markers.png"
cv2.imwrite(output_path, canvas)
print(f"网格纸已保存至: {output_path}")

# 显示图像（可选）
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("A3网格纸带ArUco标记")
plt.axis('off')
plt.show()
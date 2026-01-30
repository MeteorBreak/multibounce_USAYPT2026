# NOTES

目前的aruco纸（中间挖空那一张）的物理尺寸为322mm*152mm（四个码的右下角点形成的矩形）
    现在程序当成A3映射，待修改

待整理实验数据

导轨铁板材质的滑动摩擦因数测量

通过两个光电门的时间 ms
26.27 14.75
25.81 14.64
25.35 14.35
25.00 14.04
25.43 14.22
25.46 14.36

光电门间距 61cm

轨道倾角 58.00度

24.01 13.80
25.01 15.67
25.02 14.02

61cm
61.00度

乒乓球在亚克力
    36帧, 13度, 2105fps
    760.319度每秒
    13.27 rad/s

乒乓球在金属板 20251120 160232
    30帧, 60, 761fps
    1600.636度每秒
    27.94 rad/s

误差传播

射出高度 70.5cm

彩球 为第三个

第一个平面20*20 172 左25 25.45右
X+ -> Y+ (局部 -> 世界)
Y+ -> X-
RESULT: Frame 1391 -> X: -16.4042 mm, Y: 34.7617 mm
RESULT: Frame 2767 -> X: 20.8309 mm, Y: 53.2800 mm
RESULT: Frame 2873 -> X: 33.6229 mm, Y: 43.7583 mm
RESULT: Frame 2980 -> X: -9.5381 mm, Y: 15.4727 mm

RESULT: Frame 3194 -> X: -2.8648 mm, Y: 13.1384 mm
RESULT: Frame 3404 -> X: 4.6584 mm, Y: 41.5513 mm
RESULT: Frame 5098 -> X: 20.8882 mm, Y: 19.9258 mm
RESULT: Frame 5207 -> X: 12.8585 mm, Y: 20.0472 mm

第二个平面24*24 277 右52.5 21.6左
X+ -> Y-
Y+ -> X+
RESULT: Frame 1239 -> X: 71.9777 mm, Y: 5.1422 mm
RESULT: Frame 2615 -> X: 88.3464 mm, Y: 56.3806 mm
RESULT: Frame 2722 -> X: 62.8535 mm, Y: -5.7198 mm
RESULT: Frame 2830 -> X: 91.7079 mm, Y: 36.5631 mm

RESULT: Frame 3045 -> X: 75.7238 mm, Y: 19.2668 mm
RESULT: Frame 3254 -> X: 104.5418 mm, Y: 10.5923 mm
RESULT: Frame 4951 -> X: 25.4546 mm, Y: 37.0481 mm
RESULT: Frame 5059 -> X: 97.7193 mm, Y: 45.0714 mm

落点40*30 362 右29
X+ -> X-
Y+ -> Y-
RESULT: Frame 1109 -> X: -8.0628 mm, Y: -23.9536 mm
RESULT: Frame 2484 -> X: 12.6567 mm, Y: -35.1421 mm
RESULT: Frame 2591 -> X: -30.8733 mm, Y: 7.8142 mm
RESULT: Frame 2699 -> X: 57.9524 mm, Y: -56.7649 mm

RESULT: Frame 2914 -> X: 34.5635 mm, Y: -30.5233 mm
RESULT: Frame 3120 -> X: -16.5555 mm, Y: -68.3908 mm
RESULT: Frame 4815 -> X: 60.6139 mm, Y: 24.2484 mm
RESULT: Frame 4924 -> X: 21.4429 mm, Y: -56.6773 mm

出射目标点 172 左20

飞行时间 26/60 31/60

s0 0.033673 0.036831 0.022500
s1 0.054056 0.062020 0.034200
s2 0.071040 0.048630 0.061950

待完成：
    曲面实验
    人手投掷、发球机误差

## 流体模拟

在网格节点上离散计算
为表面设置滑移速度，计算域中挖掉小球

边界条件

概率密度函数

=======================================================
   Table Tennis Launcher Initial Velocity Calculator   
=======================================================

Default Parameters (Standard Ball):
Mass: 0.0027 kg, Radius: 0.02 m, Air Density: 1.225 kg/m^3
Launch Pos: [0. 0. 0.], Target Dist: 2.35 m

Current Nominal Velocity Guess: [10.   0.   0.1] m/s
Do you want to update parameters? (y/n) [n]: n

Enter relative path to CSV file with impact coordinates: pictures/SM_1_points_centered.csv

Columns found: ['-41.78894666666667', '-64.26811333333333']
Please specify which columns describe the impact coordinates (Y, Z).
Column for Y (Horizontal on target) [default '-41.78894666666667']: 
Column for Z (Vertical on target)   [default '-64.26811333333333']: 

Calculating inverse trajectories for 29 points...
This may take a few seconds per point depending on convergence...
Point 1: Target(40.158, -48.603) -> V0_calc([28.73, 41.10, -474.95]) Err_mag: 3.68e+01
Point 2: Target(24.973, -46.002) -> V0_calc([23.03, 29.91, -355.23]) Err_mag: 2.19e+01
Point 3: Target(19.318, -46.002) -> V0_calc([10.98, 23.44, -132.36]) Err_mag: 1.44e+01
Point 4: Target(11.264, -44.429) -> V0_calc([15.07, 12.28, -205.06]) Err_mag: 9.35e+00
Point 5: Target(-3.736, -33.421) -> V0_calc([7.89, -2.55, -77.84]) Err_mag: 2.98e+00
Point 6: Target(-35.027, -25.256) -> V0_calc([10.51, -28.37, 204.41]) Err_mag: 2.87e+01
Point 7: Target(-14.248, -12.736) -> V0_calc([2.95, -17.85, 4.55]) Err_mag: 7.70e-03
Point 8: Target(-57.895, -1.244) -> V0_calc([7.73, -86.25, 63.87]) Err_mag: 3.20e+01
Point 9: Target(46.674, -5.901) -> V0_calc([29.69, 544.65, 134.36]) Err_mag: 3.60e+00
Point 10: Target(25.158, -4.329) -> V0_calc([6.83, 72.88, 27.94]) Err_mag: 7.00e-02
Point 11: Target(14.891, -8.986) -> V0_calc([9.81, 5.78, 191.89]) Err_mag: 1.35e+01
Point 12: Target(9.973, -8.986) -> V0_calc([9.84, 10.49, 192.05]) Err_mag: 7.47e+00
Point 13: Target(7.268, -3.482) -> V0_calc([3.69, 9.76, 2.27]) Err_mag: 1.69e+00
Point 14: Target(5.916, 0.691) -> V0_calc([3.44, 8.66, 7.31]) Err_mag: 2.12e-03
Point 15: Target(-14.248, -0.397) -> V0_calc([28.81, -155.94, 667.09]) Err_mag: 2.57e+00
Point 16: Target(-6.195, 0.510) -> V0_calc([6.55, -17.77, 109.06]) Err_mag: 1.90e-01
Point 17: Target(-2.814, 8.433) -> V0_calc([3.94, -4.72, 23.39]) Err_mag: 5.43e-03
Point 18: Target(17.350, 17.022) -> V0_calc([10.30, 75.52, 104.58]) Err_mag: 1.90e-01
Point 19: Target(26.756, 18.353) -> V0_calc([17.99, 199.65, 195.90]) Err_mag: 8.93e-01
Point 20: Target(38.621, 34.623) -> V0_calc([47.03, 633.02, 698.43]) Err_mag: 1.00e+01
Point 21: Target(17.781, 29.119) -> V0_calc([22.25, 161.63, 324.80]) Err_mag: 1.50e+00
Point 22: Target(5.240, 25.187) -> V0_calc([12.57, 27.73, 165.52]) Err_mag: 3.22e-01
Point 23: Target(-0.109, 45.450) -> V0_calc([40.38, -1.61, 770.13]) Err_mag: 6.71e+00
Point 24: Target(-0.109, 60.812) -> V0_calc([60.16, -2.00, 1259.63]) Err_mag: 1.77e+01
Point 25: Target(-23.654, 30.208) -> V0_calc([28.19, -264.97, 416.49]) Err_mag: 2.77e+00
Point 26: Target(-50.703, 9.099) -> V0_calc([15.50, -207.25, 102.23]) Err_mag: 1.95e+01
Point 27: Target(-53.162, 15.450) -> V0_calc([47.68, -876.74, 441.79]) Err_mag: 1.09e+01
Point 28: Target(18.027, 28.877) -> V0_calc([22.10, 162.89, 320.87]) Err_mag: 1.48e+00
Point 29: Target(-25.682, 30.208) -> V0_calc([29.99, -303.12, 440.34]) Err_mag: 3.25e+00

========================================
              RESULTS
========================================
Nominal Velocity: [10.   0.   0.1]
Mean Calc Velocity: [ 19.4346689    1.73297713 193.91218603]
Std Dev (Dispersion): [ 14.70023698 249.97274829 344.84342819]

Velocity Errors (Calculated - Nominal):
Mean Error: [  9.4346689    1.73297713 193.81218603]
RMS Error:  [ 17.46739661 249.97875529 395.575724  ]

Save calculated velocities to CSV? (y/n): y
Saved to velocity_analysis_result.csv

=== Covariance Propagation Simulation ===
Initial Sigma R: [0.0, 0.0, 0.0]

[Step 1] Flight (0.335s) -> Pos: [0.         0.49912624 1.70260387]
  Pos Covariance Diag: [0.00050899 0.00060893 0.00022726]

>>> BOUNCE 2 IMPACT (Plot Args 2D):
    --random 5.089947e-04 0.000000e+00 0.000000e+00 2.272556e-04
    (Std dev X: 0.0226 m, Z: 0.0151 m)
  -> Bounce complete. New Vel: [1.07178602 0.97964742 2.69550676]

[Step 3] Flight (0.327s) -> Pos: [0.30971939 0.25775412 2.53108757]
  Pos Covariance Diag: [0.00107711 0.00030747 0.0005714 ]

>>> BOUNCE 4 IMPACT (Plot Args 2D):
    --random 1.077106e-03 1.519916e-20 1.519916e-20 5.714012e-04
    (Std dev X: 0.0328 m, Z: 0.0239 m)
  -> Bounce complete. New Vel: [-0.56027545  1.82725921  2.1267396 ]

[Step 5] Flight (0.299s) -> Pos: [0.15150713 0.32986864 3.14424114]
  Pos Covariance Diag: [0.00163525 0.00017964 0.00095779]

>>> BOUNCE 6 IMPACT (Plot Args 2D):
    --random 1.635254e-03 5.431297e-20 5.431297e-20 9.577918e-04
    (Std dev X: 0.0404 m, Z: 0.0309 m)
  -> Bounce complete. New Vel: [-0.2414398   0.9808518   2.07389528]

[Step 7] Flight (0.243s) -> Pos: [0.09497431 0.26319464 3.63098185]
  Pos Covariance Diag: [0.01120971 0.00453149 0.00658991]

>>> FINAL STATE (Plot Args 2D):
    --random 1.120971e-02 3.886126e-19 3.886126e-19 6.589906e-03
    (Std dev X: 0.1059 m, Z: 0.0812 m)
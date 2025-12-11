
import argparse
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import find_peaks

def find_bounce_frames(video_path: str, prominence: float = 0.05, distance_ms: int = 100):
    """
    分析视频文件，检测乒乓球弹跳声，并返回对应的帧位置。

    Args:
        video_path (str): 视频文件的路径。
        prominence (float, optional): 用于find_peaks的显著性参数。
                                    值越大，检测要求越苛刻，可能滤掉较弱的撞击声。
                                    值越小，检测越敏感，可能引入噪声。
                                    建议范围在 0.01 到 0.2 之间调整。
                                    默认为 0.05。
        distance_ms (int, optional): 两次撞击声之间的最小时间间隔（毫秒）。
                                     用于避免在同一次撞击中检测到多个峰值。
                                     默认为 100。

    Returns:
        tuple[np.ndarray, float, np.ndarray, float] | None:
            如果成功，返回一个元组，包含:
            - bounce_frames (np.ndarray): 包含撞击声的帧号数组。
            - video_fps (float): 视频的帧率。
            - audio_waveform (np.ndarray): 单声道音频波形。
            - sample_rate (float): 音频的采样率。
            如果视频无法加载或没有音轨，则返回 None。
    """
    try:
        print(f"正在加载视频文件: {video_path}...")
        video_clip = VideoFileClip(video_path)
        audio = video_clip.audio
    except Exception as e:
        print(f"错误: 无法加载或处理视频文件。 {e}")
        return None

    if audio is None:
        print("错误: 视频文件中没有音轨。")
        return None

    print("正在提取音频数据...")
    # 获取音频波形为Numpy数组
    audio_waveform = audio.to_soundarray()
    sample_rate = audio.fps
    video_fps = video_clip.fps

    # 如果是立体声，转换为单声道
    if audio_waveform.ndim > 1 and audio_waveform.shape[1] == 2:
        audio_waveform = audio_waveform.mean(axis=1)

    # 归一化并取绝对值，以便寻找能量峰值
    normalized_waveform = np.abs(audio_waveform) / np.max(np.abs(audio_waveform))

    # 将最小间隔从毫秒转换为采样点数
    min_distance_samples = int(sample_rate * (distance_ms / 1000))

    print("正在检测撞击声...")
    # 寻找峰值
    peaks, _ = find_peaks(normalized_waveform, prominence=prominence, distance=min_distance_samples)

    if len(peaks) == 0:
        print("未检测到明显的撞击声。")
        return np.array([]), video_fps, audio_waveform, sample_rate

    # 将峰值位置（采样点）转换为视频帧号
    bounce_frames = (peaks / sample_rate * video_fps).astype(int)
    
    video_clip.close()

    return bounce_frames, video_fps, audio_waveform, sample_rate

def plot_results(bounce_frames: np.ndarray, video_fps: float, audio_waveform: np.ndarray, sample_rate: float):
    """
    可视化分析结果，包括波形图和频谱图。
    """
    print("正在生成可视化图表...")
    time_axis = np.arange(len(audio_waveform)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('乒乓球撞击声检测结果', fontsize=16)

    # 1. 绘制音频波形图
    ax1.plot(time_axis, audio_waveform, label='音频波形', color='cornflowerblue', linewidth=0.5)
    ax1.set_title('音频波形和检测到的撞击位置')
    ax1.set_ylabel('振幅')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. 绘制频谱图
    ax2.specgram(audio_waveform, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    ax2.set_title('音频频谱图')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('频率 (Hz)')

    # 在两个图上标记撞击位置
    bounce_times = bounce_frames / video_fps
    for t in bounce_times:
        ax1.axvline(x=t, color='r', linestyle='--', label='撞击位置' if '撞击位置' not in [l.get_label() for l in ax1.lines] else "")
        ax2.axvline(x=t, color='r', linestyle='--')

    ax1.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("图表已生成，请查看弹出的窗口。关闭图表窗口后程序将退出。")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='从视频中检测乒乓球落地声音并标记帧位置。')
    parser.add_argument('video_path', type=str, help='输入视频文件的路径 (例如: my_video.mp4)')
    parser.add_argument('--prominence', type=float, default=0.05, help='检测峰值的显著性阈值 (0.01-0.2)。值越小越敏感。')
    parser.add_argument('--distance', type=int, default=100, help='两次撞击声之间的最小时间间隔（毫秒）。')

    args = parser.parse_args()

    results = find_bounce_frames(args.video_path, args.prominence, args.distance)

    if results is None:
        return

    bounce_frames, video_fps, audio_waveform, sample_rate = results
    
    if bounce_frames.size > 0:
        # 对帧号进行排序并去重
        unique_frames = sorted(list(set(bounce_frames)))
        print("\n--- 检测完成 ---")
        print(f"共检测到 {len(unique_frames)} 次撞击声。")
        print("识别到的帧位置:")
        # 将所有帧号格式化为一行输出
        print(', '.join(map(str, unique_frames)))
        print("------------------\n")
        
        plot_results(np.array(unique_frames), video_fps, audio_waveform, sample_rate)
    else:
        print("\n分析完成，但未找到符合条件的撞击声。")
        print("您可以尝试调整 `--prominence` 参数（例如，使用一个更小的值 `--prominence 0.02`）来提高灵敏度。")


if __name__ == '__main__':
    main()

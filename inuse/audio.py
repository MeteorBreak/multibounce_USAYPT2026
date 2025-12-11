import argparse
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.signal import find_peaks, butter, filtfilt

def high_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data.
    :param data: Audio data
    :param cutoff: Cutoff frequency (Hz)
    :param fs: Sampling rate (Hz)
    :param order: Filter order
    :return: Filtered data
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def find_bounce_frames(video_path: str, prominence: float = 0.1, distance_ms: int = 150, high_pass_cutoff_hz: int = 2000):
    """
    Analyze the video file, detect the sound of table tennis ball bouncing, and return the corresponding frame positions.

    Args:
        video_path (str): The path to the video file.
        prominence (float, optional): The prominence parameter for find_peaks.
                                    A larger value makes the detection more stringent and may filter out weaker impact sounds.
                                    A smaller value makes the detection more sensitive and may introduce noise.
                                    It is recommended to adjust between 0.01 and 0.2.
                                    Defaults to 0.05.
        distance_ms (int, optional): The minimum time interval between two impact sounds in milliseconds.
                                     Used to avoid detecting multiple peaks in the same impact.
                                     Defaults to 100.

    Returns:
        tuple[np.ndarray, float, np.ndarray, float] | None:
            If successful, returns a tuple containing:
            - bounce_frames (np.ndarray): An array of frame numbers containing the impact sounds.
            - video_fps (float): The frame rate of the video.
            - audio_waveform (np.ndarray): The mono audio waveform.
            - sample_rate (float): The sampling rate of the audio.
            If the video cannot be loaded or has no audio track, returns None.
    """
    try:
        print(f"Loading video file: {video_path}...")
        video_clip = VideoFileClip(video_path)
        audio = video_clip.audio
    except Exception as e:
        print(f"Error: Unable to load or process video file. {e}")
        return None

    if audio is None:
        print("Error: No audio track in the video file.")
        return None

    print("Extracting audio data...")
    # Get the audio waveform as a Numpy array
    audio_waveform = audio.to_soundarray()
    sample_rate = audio.fps
    video_fps = video_clip.fps

    # If it is stereo, convert to mono
    if audio_waveform.ndim > 1 and audio_waveform.shape[1] == 2:
        audio_waveform = audio_waveform.mean(axis=1)

    # Apply a high-pass filter to reduce low-frequency noise
    #print(f"Applying high-pass filter (cutoff frequency: {high_pass_cutoff_hz} Hz)...")
    filtered_waveform = high_pass_filter(audio_waveform, high_pass_cutoff_hz, sample_rate)

    # Normalize and take the absolute value to find energy peaks
    normalized_waveform = np.abs(filtered_waveform) / np.max(np.abs(filtered_waveform))

    # Convert the minimum interval from milliseconds to the number of sample points
    min_distance_samples = int(sample_rate * (distance_ms / 1000))

    print("Detecting impact sounds...")
    # Find peaks
    peaks, _ = find_peaks(normalized_waveform, prominence=prominence, distance=min_distance_samples)

    if len(peaks) == 0:
        print("No obvious impact sounds were detected.")
        return np.array([]), video_fps, audio_waveform, sample_rate

    # Convert the peak position (sample point) to the video frame number
    bounce_frames = (peaks / sample_rate * video_fps).astype(int)
    
    video_clip.close()

    return bounce_frames, video_fps, audio_waveform, sample_rate

def plot_results(bounce_frames: np.ndarray, video_fps: float, audio_waveform: np.ndarray, sample_rate: float):
    """
    Visualize the analysis results, including waveform and spectrogram.
    """
    print("Generating visualization charts...")
    time_axis = np.arange(len(audio_waveform)) / sample_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Ping-pong Ball Impact Sound Detection Results', fontsize=16)

    # 1. Plot the audio waveform
    ax1.plot(time_axis, audio_waveform, label='Audio Waveform', color='cornflowerblue', linewidth=0.5)
    ax1.set_title('Audio Waveform and Detected Impact Positions')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Plot the spectrogram
    ax2.specgram(audio_waveform, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='viridis')
    ax2.set_title('Audio Spectrogram')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    # Mark the impact positions on both plots
    bounce_times = bounce_frames / video_fps
    for t in bounce_times:
        ax1.axvline(x=t, color='r', linestyle='--', label='Impact Position' if 'Impact Position' not in [l.get_label() for l in ax1.lines] else "")
        ax2.axvline(x=t, color='r', linestyle='--')

    ax1.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Chart has been generated, please check the pop-up window. The program will exit after closing the chart window.")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detect the sound of a ping-pong ball landing in a video and mark the frame position.')
    parser.add_argument('video_path', type=str, help='Path to the input video file (e.g., my_video.mp4)')
    parser.add_argument('--prominence', type=float, default=0.1, help='Prominence threshold for peak detection (0.01-0.2). Smaller values are more sensitive.')
    parser.add_argument('--distance', type=int, default=150, help='Minimum time interval between two impact sounds in milliseconds.')
    parser.add_argument('--cutoff', type=int, default=2000, help='Cutoff frequency for the high-pass filter (Hz). Used to filter out low-frequency noise.')

    args = parser.parse_args()

    results = find_bounce_frames(args.video_path, args.prominence, args.distance, args.cutoff)

    if results is None:
        return

    bounce_frames, video_fps, audio_waveform, sample_rate = results
    
    if bounce_frames.size > 0:
        # Sort and remove duplicate frame numbers
        unique_frames = sorted(list(set(bounce_frames)))
        print("\n--- Detection Complete ---")
        print(f"A total of {len(unique_frames)} impact sounds were detected.")
        print("Recognized frame positions:")
        # Format all frame numbers into one line of output
        print(', '.join(map(str, unique_frames)))
        print("------------------\n")
        
        plot_results(np.array(unique_frames), video_fps, audio_waveform, sample_rate)
    else:
        print("\nAnalysis complete, but no qualifying impact sounds were found.")
        print("You can try adjusting the `--prominence` parameter (e.g., use a smaller value like `--prominence 0.02`) to increase sensitivity.")


if __name__ == '__main__':
    main()

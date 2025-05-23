import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ffmpeg_quality_metrics import FfmpegQualityMetrics

# Hàm lấy danh sách video
def get_video_files(folder_path):
    video_extensions = ('*.mp4', '*.avi', '*.mkv', '*.mov')
    files = []
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    return [os.path.basename(f) for f in files]

# Hàm tính VMAF
def calculate_vmaf(reference_video, distorted_video):
    try:
        ffqm = FfmpegQualityMetrics(reference_video, distorted_video)
        metrics = ffqm.calculate(["vmaf"])
        vmaf_scores = [frame['vmaf'] for frame in metrics['vmaf']]  
        return vmaf_scores
    except Exception as e:
        print(f"Lỗi khi tính VMAF: {e}")
        return None

# Các đường dẫn thư mục
folder_source_path = './video_source_full/'
normal_video_prefix = './video_output/normal_BW_'
gai_video_prefix = './video_output/GAI_BW_'

target_size_list = [0.05, 0.1, 0.2, 0.5]

# Lấy danh sách video
video_list = get_video_files(folder_source_path)

# Lưu VMAF cho từng target size
vmaf_scores_by_target = {}

for target_size in target_size_list:
    normal_vmaf_scores = []
    gai_vmaf_scores = []

    for video_name in video_list:
        ref_vid = os.path.join(folder_source_path, video_name)
        normal_vid = os.path.join(f"{normal_video_prefix}{target_size}_{video_name}")
        gai_vid = os.path.join(f"{gai_video_prefix}{target_size}_{video_name}")

        if os.path.exists(normal_vid) and os.path.exists(gai_vid):
            print(f'Processing video: {video_name} at target_size {target_size}')
            normal_mos = calculate_vmaf(ref_vid, normal_vid)
            gai_mos = calculate_vmaf(ref_vid, gai_vid)

            if normal_mos is None or gai_mos is None:
                print(f"Bỏ qua video {video_name} do lỗi xử lý.")
                continue

            normal_vmaf_scores.extend(normal_mos)
            gai_vmaf_scores.extend(gai_mos)
        else:
            print(f'Skipping {video_name} (missing files) for target_size {target_size}')

    vmaf_scores_by_target[target_size] = {
        "normal": normal_vmaf_scores,
        "gai": gai_vmaf_scores
    }




# Plot CDF cho từng target_size
plt.figure(figsize=(10, 7))
for target_size, scores in vmaf_scores_by_target.items():
    if scores["normal"] and scores["gai"]:
        # Sort and calculate CDF
        normal_sorted = np.sort(scores["normal"])
        gai_sorted = np.sort(scores["gai"])
        cdf_normal = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
        cdf_gai = np.arange(1, len(gai_sorted) + 1) / len(gai_sorted)

        plt.plot(normal_sorted, cdf_normal, linestyle='-', label=f'Normal {target_size}')
        plt.plot(gai_sorted, cdf_gai, linestyle='--', label=f'GAI {target_size}')

plt.xlabel('VMAF')
plt.ylabel('CDF')
plt.title('CDF of VMAF Scores for Different Target Sizes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
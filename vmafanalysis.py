#%%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from ffmpeg_quality_metrics import FfmpegQualityMetrics

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


folder_source_path = '/path/to/source_vid/'
source_file = "received_video_fps24.mp4"

normal_video_prefix = "path/to/abr" 
gai_video_prefix = "/path/to/gai" 
nogai_video_prefix = "/path/to/nogai"

target_size_list = [0.05, 0.1, 0.2, 0.5]
target_size_list = [0.25]

video_list = ['received_video_fps24_1.mp4','received_video_fps24_2.mp4','received_video_fps24_3.mp4','received_video_fps24_4.mp4','received_video_fps24_5.mp4',
              'received_video_fps24_6.mp4','received_video_fps24_7.mp4','received_video_fps24_8.mp4','received_video_fps24_9.mp4','received_video_fps24_10.mp4'] 
# This is a list of the received videos names that would be run through after generating data



# Lưu VMAF cho từng target size
vmaf_scores_by_target = {}

for target_size in target_size_list:
    normal_vmaf_scores = []
    gai_vmaf_scores = []
    nogai_vmaf_scores = []

    for video_name in video_list:
        ref_vid = os.path.join(folder_source_path, source_file)
        normal_vid = os.path.join(f"{normal_video_prefix}{target_size}_{video_name}")
        gai_vid = os.path.join(f"{gai_video_prefix}{target_size}_{video_name}")
        nogai_vid = os.path.join(f"{nogai_video_prefix}{target_size}_{video_name}")
        if os.path.exists(normal_vid) and os.path.exists(gai_vid) and os.path.exists(nogai_vid):
            print(f'Processing video: {video_name} at target_size {target_size}')
            normal_mos = calculate_vmaf(ref_vid, normal_vid)
            print("processing abr")

            gai_mos = calculate_vmaf(ref_vid, gai_vid)
            print("processing gai")
            nogai_mos = calculate_vmaf(ref_vid, nogai_vid)
            print("processing nogai")

            if normal_mos is None or gai_mos is None or nogai_mos is None:
                print(f"Bỏ qua video {video_name} do lỗi xử lý.")
                continue

            normal_vmaf_scores.extend(normal_mos)
            gai_vmaf_scores.extend(gai_mos)
            nogai_vmaf_scores.extend(nogai_mos)

        else:
            print(f'Skipping {video_name} (missing files) for target_size {target_size}')
    print("done")
    vmaf_scores_by_target[target_size] = {
        "normal": normal_vmaf_scores,
        "gai": gai_vmaf_scores,
        "nogai":nogai_vmaf_scores
    }

# Plot CDF cho từng target_size
plt.figure(figsize=(10, 7))
for target_size, scores in vmaf_scores_by_target.items():
    if scores["normal"] and scores["gai"] and scores['nogai']:
        # Sort and calculate CDF
        normal_sorted = np.sort(scores["normal"])
        gai_sorted = np.sort(scores["gai"])
        nogai_sorted = np.sort(scores["nogai"])

        cdf_normal = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
        cdf_gai = np.arange(1, len(gai_sorted) + 1) / len(gai_sorted)
        cdf_nogai = np.arange(1, len(nogai_sorted) + 1) / len(nogai_sorted)

        plt.plot(normal_sorted, cdf_normal, label=f'Normal {target_size}')
        plt.plot(gai_sorted, cdf_gai, label=f'GAI {target_size}')
        plt.plot(nogai_sorted, cdf_nogai, label=f'no GAI {target_size}')


import pickle
print("dumping normal")
with open('normal_sorted.pkl', 'wb') as f:
    pickle.dump(scores["normal"],f)
    
print("dumping gai")
with open('gai_sorted.pkl', 'wb') as f:
    pickle.dump(scores["gai"],f)

print("dumping nogai")
with open('nogai_sorted.pkl', 'wb') as f:
    pickle.dump(scores["nogai"],f)

plt.xlabel('VMAF')
plt.ylabel('CDF')
plt.legend()
plt.savefig(f"Figures/results/vmaf_comp")
plt.show()


import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 从文件中读取数据
def read_data_from_file(file_path, file_prefix, max_frames=50):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            time, person_id, x, y = map(float, line.strip().split())
            person_id = f"{file_prefix}_{int(person_id)}"  # 确保person_id唯一性
            if person_id not in data:
                data[person_id] = {'time': [], 'x': [], 'y': []}
            data[person_id]['time'].append(time)
            data[person_id]['x'].append(x)
            data[person_id]['y'].append(y)

            # 限制为前50帧
            if len(data[person_id]['time']) == max_frames:
                break
    return data

# 从文件夹中读取数据
def read_data_from_folder(folder_path, max_frames=50):
    all_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_prefix = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            file_data = read_data_from_file(file_path, file_prefix, max_frames)
            all_data.update(file_data)
    return all_data

# 检测缺帧
def detect_missing_frames(data):
    missing_frames_info = {}
    for person_id, trajectory in data.items():
        times = np.array(trajectory['time'])
        total_frames = len(times)
        if total_frames < 2:
            continue
        time_diffs = np.diff(times)
        missing_frames = np.sum(time_diffs > 1)
        if missing_frames > 0:
            missing_frames_info[person_id] = {
                'total_frames': total_frames,
                'missing_frames': missing_frames,
                'frame_percentage': (missing_frames / total_frames) * 100
            }
    return missing_frames_info

# 分析数据
def analyze_data(folders, max_frames=50):
    all_data = {}
    for folder in folders:
        data = read_data_from_folder(folder, max_frames)
        all_data.update(data)
    
    missing_frames_info = detect_missing_frames(all_data)
    
    total_trajectories = len(all_data)
    total_missing_frames_trajectories = len(missing_frames_info)
    missing_frames_percentage = (total_missing_frames_trajectories / total_trajectories) * 100
    small_trajectories_count = sum(1 for info in missing_frames_info.values() if info['total_frames'] <= 10)
    
    large_trajectories_info = {k: v for k, v in missing_frames_info.items() if v['total_frames'] > 10}
    missing_frames_percentages = [info['frame_percentage'] for info in large_trajectories_info.values()]

    # 打印统计信息
    print(f"Total number of trajectories: {total_trajectories}")
    print(f"Number of trajectories with missing frames: {total_missing_frames_trajectories}")
    print(f"Percentage of trajectories with missing frames: {missing_frames_percentage:.2f}%")
    print(f"Number of trajectories with total frames <= 10: {small_trajectories_count}")
    
    # 使用Seaborn绘制直方图
    plot_histogram(missing_frames_percentages)

# 绘制直方图
def plot_histogram(missing_frames_percentages):
    plt.figure(figsize=(10, 6))
    
    # 设置更多的分段来确保横坐标信息足够详细
    bins = np.arange(0, 101, 5)  # 以每5%为一段的分段，确保横坐标显示更细
    sns.histplot(missing_frames_percentages, bins=bins, kde=False, color="skyblue")

    plt.title('Distribution of Missing Frame Percentages (First 50 Frames)', fontsize=16)
    plt.xlabel('Missing Frame Percentage (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # 标注每个柱子的频数
    counts, bins_edges = np.histogram(missing_frames_percentages, bins=bins)
    for i in range(len(counts)):
        plt.text(bins_edges[i] + (bins_edges[i+1] - bins_edges[i]) / 2, counts[i], f'{counts[i]}', 
                 ha='center', va='bottom', fontsize=10, rotation=90)
    
    # 确保每个柱子都有对应的横坐标显示
    plt.xticks(bins)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # 保存为图片
    plt.savefig('/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/missing_frames_distribution_50frames_refined.png')
    plt.show()


if __name__ == "__main__":
    folders = [
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/val/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/test/'
    ]
    analyze_data(folders)

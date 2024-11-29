'''
import os
import numpy as np
import matplotlib.pyplot as plt
import csv

def read_data_from_file(file_path, file_prefix):
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
    return data

def read_data_from_folder(folder_path):
    all_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_prefix = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            file_data = read_data_from_file(file_path, file_prefix)
            all_data.update(file_data)
    return all_data

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

def analyze_data(folders):
    all_data = {}
    for folder in folders:
        data = read_data_from_folder(folder)
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
    
    # 保存直方图数据为CSV格式
    csv_file = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/missing_frames_data.csv'
    bins = np.arange(0, 101, 2)
    counts, _ = np.histogram(missing_frames_percentages, bins=bins)
    
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Bin Start', 'Bin End', 'Frequency'])
        for i in range(len(bins) - 1):
            writer.writerow([bins[i], bins[i+1], counts[i]])
    
    print(f"Histogram data saved to {csv_file}")

if __name__ == "__main__":
    folders = [
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/val/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/test/'
    ]
    analyze_data(folders)
'''
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_data_from_file(file_path, file_prefix):
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
    return data

def read_data_from_folder(folder_path):
    all_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_prefix = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            file_data = read_data_from_file(file_path, file_prefix)
            all_data.update(file_data)
    return all_data

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

def analyze_data(folders):
    all_data = {}
    for folder in folders:
        data = read_data_from_folder(folder)
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
    print(f"Percentage of trajectories with total frames <= 10: {(small_trajectories_count/total_trajectories) * 100:.2f}%")
    
    # 使用Seaborn绘制直方图
    plot_histogram(missing_frames_percentages)

def plot_histogram(missing_frames_percentages):
    plt.figure(figsize=(10, 6))
    
    # 使用Seaborn的直方图绘制
    sns.histplot(missing_frames_percentages, bins=30, kde=True, color="skyblue")
    
    plt.title('Distribution of Missing Frame Percentages', fontsize=16)
    plt.xlabel('Missing Frame Percentage (%)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # 标注每个柱子的频数
    counts, bins = np.histogram(missing_frames_percentages, bins=30)
    for i in range(len(counts)):
        plt.text(bins[i] + (bins[i+1] - bins[i]) / 2, counts[i], f'{counts[i]}', 
                 ha='center', va='bottom', fontsize=10, rotation=90)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # 保存为图片
    plt.savefig('/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/vehicle_missing_frames_distribution.png')
    plt.show()

if __name__ == "__main__":
    folders = [
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_vehicle/train/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_vehicle/val/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_vehicle/test/'
    ]
    analyze_data(folders)


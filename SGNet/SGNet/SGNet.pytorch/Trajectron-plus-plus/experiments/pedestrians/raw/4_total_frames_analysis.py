import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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

def process_single_folder(args):
    folder_path, filenames = args
    all_data = {}
    for filename in filenames:
        if filename.endswith('.txt'):
            file_prefix = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            file_data = read_data_from_file(file_path, file_prefix)
            all_data.update(file_data)
    return all_data

def read_data_from_folders(folders):
    all_data = {}
    
    # 准备参数，每个文件夹分配多个文件
    folder_tasks = [(folder, os.listdir(folder)) for folder in folders]
    
    with Pool(processes=cpu_count()) as pool:
        results = []
        for task in tqdm(folder_tasks, desc="Processing Folders"):
            result = pool.apply_async(process_single_folder, (task,))
            results.append(result)
        
        for result in tqdm(results, desc="Merging Data"):
            data = result.get()
            all_data.update(data)
    
    return all_data

def count_total_frames(data):
    frame_counts = []
    for person_id, trajectory in data.items():
        total_frames = len(trajectory['time'])
        frame_counts.append(total_frames)
    return frame_counts

def analyze_data(folders):
    # 读取所有文件夹的数据
    all_data = read_data_from_folders(folders)
    
    # 统计每个行人的总帧数
    total_frame_counts = count_total_frames(all_data)
    
    # 打印统计信息
    print(f"Total number of trajectories: {len(total_frame_counts)}")
    print(f"Average frame count per trajectory: {np.mean(total_frame_counts):.2f}")
    print(f"Maximum frame count: {np.max(total_frame_counts)}")
    print(f"Minimum frame count: {np.min(total_frame_counts)}")
    
    # 绘制总帧数分布的直方图
    plot_histogram(total_frame_counts)

def plot_histogram(total_frame_counts):
    plt.figure(figsize=(10, 6))
    
    # 使用Seaborn的直方图绘制
    sns.histplot(total_frame_counts, bins=30, kde=True, color="skyblue")
    
    plt.title('Distribution of Total Frame Counts', fontsize=16)
    plt.xlabel('Total Frame Count', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    
    # 标注每个柱子的频数
    counts, bins = np.histogram(total_frame_counts, bins=30)
    for i in range(len(counts)):
        plt.text(bins[i] + (bins[i+1] - bins[i]) / 2, counts[i], f'{counts[i]}', 
                 ha='center', va='bottom', fontsize=10, rotation=90)
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # 保存为图片
    plt.savefig('/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/infra_total_frames_distribution.png')
    plt.show()

if __name__ == "__main__":
    folders = [
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/val/',
        '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/test/'
    ]
    analyze_data(folders)

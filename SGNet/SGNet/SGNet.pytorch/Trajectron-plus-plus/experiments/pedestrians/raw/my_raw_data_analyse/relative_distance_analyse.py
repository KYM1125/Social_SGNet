import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance_matrix

def calculate_interaction_metrics(group, threshold_distance):
    """计算交互行为指标，如碰撞频率和相互靠近的时间"""
    n = len(group)
    if n < 2:
        return 0, 0
    
    # 获取所有行人的坐标和时间
    x = group['x'].values
    y = group['y'].values
    time = group['time'].values
    
    # 计算每对行人之间的距离矩阵
    coords = np.vstack((x, y)).T
    dist_matrix = distance_matrix(coords, coords)
    
    # 计算碰撞频率和相互靠近的时间
    collision_count = np.sum(dist_matrix < threshold_distance)
    close_time = np.sum(dist_matrix < threshold_distance) / (n * (n - 1))  # Normalize by number of pairs
    
    return collision_count, close_time

def analyze_interactions(folder_path, threshold_distance):
    all_data = []

    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, delimiter='\t', header=None, names=['time', 'person_id', 'x', 'y'])

            # 添加文件名作为前缀以确保person_id唯一
            data['person_id'] = data['person_id'].apply(lambda x: f"{file_name}_{int(x)}")
            all_data.append(data)
    
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 计算交互行为指标
    grouped = combined_data.groupby('person_id')
    
    collision_counts = []
    close_times = []
    
    for _, group in grouped:
        if len(group) < 2:
            continue
        
        # 排序时间
        group = group.sort_values(by='time')
        
        # 计算交互行为指标
        collision_count, close_time = calculate_interaction_metrics(group, threshold_distance)
        collision_counts.append(collision_count)
        close_times.append(close_time)
    
    # 绘制交互行为指标的直方图
    plt.figure(figsize=(12, 6))

    # 碰撞次数直方图
    plt.subplot(1, 2, 1)
    plt.hist(collision_counts, bins=20, edgecolor='black', alpha=0.7)
    plt.yscale('log')  # 对数刻度
    plt.xlabel('Collision Count')
    plt.ylabel('Frequency')
    plt.title('Histogram of Collision Counts')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(close_times, bins=30, edgecolor='black')
    plt.xlabel('Close Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Close Times')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/ubuntu220403/pedestrain trajetory prediction/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians/raw/raw/all_data'
    threshold_distance = 1.0  # Define your threshold distance for interactions
    analyze_interactions(folder_path, threshold_distance)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler


def calculate_average_acceleration(x, y, time):
    """计算轨迹的平均加速度"""
    velocities = []
    accelerations = []

    # 计算速度
    for i in range(1, len(time)):
        dx = x[i] - x[i-1]
        dy = y[i] - y[i-1]
        dt = (time[i] - time[i-1]) / 10.0
        if dt > 0:
            velocities.append(np.sqrt(dx**2 + dy**2) / dt)
    # 对速度进行平滑处理
    # velocities = gaussian_filter1d(velocities, sigma=1.0)
    # 计算加速度
    for i in range(1, len(velocities)):
        dv = velocities[i] - velocities[i-1]
        dt = (time[i] - time[i-1]) / 10.0
        if dt > 0:
            accelerations.append(dv / dt)

    avg_acceleration = np.mean(accelerations) if accelerations else np.nan
    return avg_acceleration

def analyze_trajectory_data(folder_path):
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
    
    # 计算平均加速度
    grouped = combined_data.groupby('person_id')
    
    average_accelerations = []
    
    for person_id, group in grouped:
        if len(group) < 3:
            continue
        
        # 排序时间
        group = group.sort_values(by='time')
        x = group['x'].values
        y = group['y'].values
        time = group['time'].values
        
        # 计算平均加速度
        avg_acceleration = calculate_average_acceleration(x, y, time)
        average_accelerations.append(avg_acceleration)

        # 对加速度进行标准化
        # scaler = StandardScaler()
        # normalized_accelerations = scaler.fit_transform(np.array(average_accelerations).reshape(-1, 1)).flatten()
    
    # 绘制平均加速度的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(average_accelerations, bins=200, edgecolor='black')
    plt.xlabel('Average Acceleration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Accelerations')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/ubuntu220403/pedestrain trajetory prediction/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians/raw/raw/all_data'
    analyze_trajectory_data(folder_path)

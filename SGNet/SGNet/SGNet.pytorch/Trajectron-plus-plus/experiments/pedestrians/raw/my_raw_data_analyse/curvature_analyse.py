import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

def calculate_curvature(x, y):
    """计算曲线的曲率"""
    n = len(x)
    if n < 3:
        return np.nan
    
    # 计算曲率的近似值
    curvature = np.zeros(n)
    for i in range(1, n-1):
        x1, x2, x3 = x[i-1], x[i], x[i+1]
        y1, y2, y3 = y[i-1], y[i], y[i+1]

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2
        d2x = dx2 - dx1
        d2y = dy2 - dy1
        denominator = (dx1**2 + dy1**2) * (dx2**2 + dy2**2)
        if denominator == 0:
            curvature[i] = 0
        else:
            curvature[i] = abs(d2x * dy1 - d2y * dx1) / denominator**0.5
    return curvature[1:-1]  # 返回中间部分的曲率

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
    
    # 计算曲率
    grouped = combined_data.groupby('person_id')
    
    curvatures = []
    
    for person_id, group in grouped:
        if len(group) < 3:
            continue
        
        # 排序时间
        group = group.sort_values(by='time')
        x = group['x'].values
        y = group['y'].values
        
        # 计算曲率
        curvature = calculate_curvature(x, y)
        if len(curvature) > 0:
            curvatures.append(np.mean(curvature))  # 使用曲率的均值作为特征
    
    # 绘制曲率的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(curvatures, bins=30, edgecolor='black')
    plt.xlabel('Average Curvature')
    plt.ylabel('Frequency')
    plt.title('Histogram of Average Curvatures')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/ubuntu220403/pedestrain trajetory prediction/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians/raw/raw/all_data'
    analyze_trajectory_data(folder_path)

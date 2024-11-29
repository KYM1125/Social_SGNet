import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_angle(x1, y1, x2, y2):
    """计算从点(x1, y1)到点(x2, y2)的朝向角度（以度为单位）"""
    return np.arctan2(y2 - y1, x2 - x1) * (180 / np.pi)

def calculate_angle_change(start_angle, end_angle):
    """计算角度变化，考虑正负角度"""
    angle_change = end_angle - start_angle
    if angle_change > 180:
        angle_change -= 360
    elif angle_change < -180:
        angle_change += 360
    return angle_change

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
    
    # 计算起始和最终坐标
    grouped = combined_data.groupby('person_id')
    
    angle_changes = []
    
    for person_id, group in grouped:
        if len(group) < 2:
            continue
        
        # 起始朝向（从第一帧到第二帧）
        start_x1, start_y1 = group.iloc[0][['x', 'y']]
        start_x2, start_y2 = group.iloc[1][['x', 'y']]
        start_angle = calculate_angle(start_x1, start_y1, start_x2, start_y2)
        
        # 最终朝向（从倒数第二帧到最后一帧）
        end_x1, end_y1 = group.iloc[-2][['x', 'y']]
        end_x2, end_y2 = group.iloc[-1][['x', 'y']]
        end_angle = calculate_angle(end_x1, end_y1, end_x2, end_y2)
        
        # 计算角度变化
        angle_change = calculate_angle_change(start_angle, end_angle)
        angle_changes.append(angle_change)
    
    # 绘制角度变化的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(angle_changes, bins=30, edgecolor='black')
    plt.xlabel('Angle Change (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Angle Changes')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/ubuntu220403/pedestrain trajetory prediction/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians/raw/raw/all_data'
    analyze_trajectory_data(folder_path)

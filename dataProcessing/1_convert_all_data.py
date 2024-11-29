'''
    将DAIR格式数据转换成ETH格式数据
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import numpy as np

def process_csv_file(filepath, frame_interval=0.1):
    """处理 CSV 文件并转换数据格式，同时插值缺失帧"""
    df = pd.read_csv(filepath, delimiter=',')
    
    # 筛选出行人数据
    pedestrian_data = df[df['sub_type'] == 'PEDESTRIAN']

    # 如果没有行人数据，返回空列表
    if pedestrian_data.empty:
        return []

    # 按 id 进行分组
    grouped = pedestrian_data.groupby('id')
    # grouped_all = df.groupby('id')

    # 存储转换后的数据
    eth_format_data = []
    for ped_id, group in grouped:

        # 按 timestamp 排序
        sorted_group = group.sort_values(by='timestamp').reset_index(drop=True)
        # sorted_group_all = grouped_all.sort_values(by='timestamp').reset_index(drop=True)
        # base_time = sorted_group['timestamp'].iloc[0]
        base_time = df['timestamp'].min()
        # print("base_time = ",base_time)

        # 存储插值后的数据
        interpolated_data = []

        # 遍历每一行进行插值
        for idx in range(len(sorted_group) - 1):
            row = sorted_group.iloc[idx]
            next_row = sorted_group.iloc[idx + 1]

            # 添加当前行数据
            interpolated_data.append(row)

            # 当前行与下一行之间的时间差
            time_diff = next_row['timestamp'] - row['timestamp']

            # 如果时间差大于期望的帧间隔，则插值
            missing_frames = int(time_diff / frame_interval)
            if missing_frames > 1:
                for frame in range(1, missing_frames):
                    # 新时间戳
                    new_timestamp = row['timestamp'] + frame * frame_interval

                    # 插值计算 x 和 y
                    new_x = np.interp(new_timestamp, [row['timestamp'], next_row['timestamp']], 
                                      [row['x'], next_row['x']])
                    new_y = np.interp(new_timestamp, [row['timestamp'], next_row['timestamp']], 
                                      [row['y'], next_row['y']])

                    # 创建插值行
                    interpolated_row = row.copy()
                    interpolated_row['timestamp'] = new_timestamp
                    interpolated_row['x'] = new_x
                    interpolated_row['y'] = new_y
                    interpolated_data.append(interpolated_row)

        # 添加最后一行数据
        last_row = sorted_group.iloc[-1]
        interpolated_data.append(last_row)

        # 转换为 ETH 格式
        for row in interpolated_data:
            eth_time = convert_to_eth_time(row['timestamp'], base_time)
            eth_format_data.append(f"{eth_time:.1f}\t{ped_id:.1f}\t{row['x']:.9f}\t{row['y']:.9f}")

    return eth_format_data

def convert_to_eth_time(timestamp, base_time, interval=0.1):
    """将时间戳转换为 ETH 格式时间"""
    return (timestamp - base_time) / interval


def save_to_txt(eth_data, output_filepath):
    """将转换后的数据保存到 TXT 文件"""
    with open(output_filepath, 'w') as file:
        for line in eth_data:
            file.write(line + '\n')


def process_single_file(args):
    """多进程辅助函数"""
    filepath, output_folder = args
    eth_data = process_csv_file(filepath)
    if eth_data:
        output_filename = os.path.splitext(os.path.basename(filepath))[0] + '.txt'
        output_filepath = os.path.join(output_folder, output_filename)
        save_to_txt(eth_data, output_filepath)
    else:
        print(f"No pedestrian data in {os.path.basename(filepath)}, skipped.")


def process_all_csv_files(base_input_folder, input_folders, output_folder):
    """处理指定文件夹中的所有 CSV 文件并保存到同一输出文件夹（使用多进程并行）"""
    
    # 如果目标文件夹已存在，则删除
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    
    # 收集所有子文件夹中的 CSV 文件路径
    csv_files = []
    for folder in input_folders:
        input_folder = os.path.join(base_input_folder, folder)
        csv_files.extend([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')])

    # 使用多进程加速处理
    num_workers = min(cpu_count(), len(csv_files))
    with Pool(num_workers) as pool:
        pool.map(process_single_file, [(filepath, output_folder) for filepath in csv_files])

# 输入和输出主文件夹路径
base_input_folder = '../../DAIR/V2X-Seq-TFD/cooperative-vehicle-infrastructure/infrastructure-trajectories/'
output_folder = '../../SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/all_converted'

# 要处理的子文件夹
input_folders = ['train', 'val']

process_all_csv_files(base_input_folder, input_folders, output_folder)

'''
    对缺帧进行插值
'''
import os
import numpy as np
from tqdm import tqdm
import shutil
from multiprocessing import Pool

# 读取文件并存储数据
def read_data_from_file(file_path):
    data = {}
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]  # 从文件名生成唯一前缀
    with open(file_path, 'r') as file:
        for line in file:
            time, person_id, x, y = map(float, line.strip().split())
            person_id = int(person_id)
            unique_person_id = f"{file_prefix}_{person_id}"  # 使用文件名前缀
            if unique_person_id not in data:
                data[unique_person_id] = {'time': [], 'x': [], 'y': []}
            data[unique_person_id]['time'].append(time)
            data[unique_person_id]['x'].append(x)
            data[unique_person_id]['y'].append(y)
    return data

# 根据缺失帧数逐一填补缺帧数据
def interpolate_missing_frames(trajectory, frame_interval=1):
    times = np.array(trajectory['time'])
    x_coords = np.array(trajectory['x'])
    y_coords = np.array(trajectory['y'])

    interpolated_times = [times[0]]
    interpolated_x = [x_coords[0]]
    interpolated_y = [y_coords[0]]

    # 逐一处理相邻帧间的缺失帧
    for i in range(1, len(times)):
        time_diff = times[i] - times[i - 1]

        # 计算缺失帧数
        if time_diff > frame_interval:
            missing_frames = int(time_diff // frame_interval)
            for frame in range(1, missing_frames + 1):
                new_timestamp = times[i - 1] + frame * frame_interval
                new_x = np.interp(new_timestamp, [times[i - 1], times[i]], [x_coords[i - 1], x_coords[i]])
                new_y = np.interp(new_timestamp, [times[i - 1], times[i]], [y_coords[i - 1], y_coords[i]])
                interpolated_times.append(new_timestamp)
                interpolated_x.append(new_x)
                interpolated_y.append(new_y)

        # 添加当前时间点
        interpolated_times.append(times[i])
        interpolated_x.append(x_coords[i])
        interpolated_y.append(y_coords[i])

    return np.array(interpolated_times), np.array(interpolated_x), np.array(interpolated_y)

# 保存数据
def save_data(data, output_file):
    if not data:
        return
    with open(output_file, 'w') as f:
        for person_id, traj in data.items():
            original_person_id = person_id.split('_', 1)[-1]
            for time, x, y in zip(traj['time'], traj['x'], traj['y']):
                f.write(f"{int(time)}\t{original_person_id}\t{x:.9f}\t{y:.9f}\n")

# 处理单个文件
def process_file(args):
    input_file, output_file_interpolated = args
    data = read_data_from_file(input_file)

    interpolated_data = {}
    for person_id, traj in data.items():
        times, x_coords, y_coords = interpolate_missing_frames(traj)
        interpolated_data[person_id] = {'time': times, 'x': x_coords, 'y': y_coords}

    save_data(interpolated_data, output_file_interpolated)

# 处理文件夹中的所有文件
def process_folder(input_folder, output_folder_interpolated):
    # 删除已有的 output 文件夹并重新创建
    if os.path.exists(output_folder_interpolated):
        shutil.rmtree(output_folder_interpolated)
    os.makedirs(output_folder_interpolated)

    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    file_paths = [(os.path.join(input_folder, filename), os.path.join(output_folder_interpolated, filename)) for filename in files]

    # 使用多进程处理文件
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, file_paths), total=len(file_paths), desc=f"Processing {input_folder}", unit="file"))

def main():
    input_folder = '../../SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/all_converted/'
    output_folder_interpolated = '../../SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/interpolated_all/'

    process_folder(input_folder, output_folder_interpolated)

if __name__ == "__main__":
    main()

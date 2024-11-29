import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 读取文件并存储数据
def read_data_from_file(file_path, max_frames=50):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            time, person_id, x, y = map(float, line.strip().split())
            person_id = int(person_id)  # 使用person_id作为键
            if person_id not in data:
                data[person_id] = {'time': [], 'x': [], 'y': []}
            if len(data[person_id]['time']) < max_frames:  # 仅保留前max_frames帧数据
                data[person_id]['time'].append(time)
                data[person_id]['x'].append(x)
                data[person_id]['y'].append(y)
    return data

# 插值缺帧数据
def interpolate_missing_frames(trajectory):
    times = np.array(trajectory['time'])
    x_coords = np.array(trajectory['x'])
    y_coords = np.array(trajectory['y'])

    # 插值处理 - 线性插值代替NAOMI方法
    all_times = np.arange(times.min(), times.max() + 1)
    f_x = interp1d(times, x_coords, kind='linear', fill_value="extrapolate")
    f_y = interp1d(times, y_coords, kind='linear', fill_value="extrapolate")

    interpolated_x = f_x(all_times)
    interpolated_y = f_y(all_times)

    return all_times, interpolated_x, interpolated_y

# 可视化轨迹并保存图片
def visualize_trajectory(person_id, times, x_coords, y_coords, original_times, output_folder):
    plt.figure()
    
    # 创建布尔索引
    max_time = max(times)
    min_time = max_time - 50
    time_mask = (original_times <= max_time) & (original_times >= min_time)
    
    # 确保布尔索引与数据维度匹配
    valid_indices = np.isin(times, original_times[time_mask])
    
    # 绘制前50帧的数据（绿色圆圈）
    plt.plot(np.array(x_coords)[valid_indices], np.array(y_coords)[valid_indices], 
            'go', label="Original Points within 50 Frames", markersize=10)  # 绿色圆圈
    
    # 绘制原始点（实心点）
    plt.plot(x_coords[np.isin(times, original_times)], 
             y_coords[np.isin(times, original_times)], 
             'bo', label="Original Points")  # 实心蓝色点
    
    # 绘制插值点（空心点）
    plt.plot(x_coords[~np.isin(times, original_times)], 
             y_coords[~np.isin(times, original_times)], 
             'ro', markerfacecolor='none', label="Interpolated Points")  # 空心红色点
    
    # 绘制连接线
    plt.plot(x_coords, y_coords, 'b-', alpha=0.5)  # 连接线（蓝色）

    plt.title(f"Interpolated Trajectory for Person {person_id}")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.legend()
    plt.grid(True)

    # 保存图片
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, f"person_{person_id}_trajectory.png"))
    plt.close()


# 删除少于等于3帧的行人
def filter_short_trajectories(data, min_frames=3):
    filtered_data = {person_id: traj for person_id, traj in data.items() if len(traj['time']) > min_frames}
    return filtered_data

# 删除插值后帧数小于等于3帧的行人
def filter_small_trajectories(data):
    filtered_data = {}
    for person_id, traj in data.items():
        times = np.array(traj['time'])
        if len(times) > 3:  # 如果插值后的帧数大于3，保留该行人数据
            filtered_data[person_id] = traj
    return filtered_data

# 保存处理后的数据
def save_data(data, output_file):
    with open(output_file, 'w') as f:
        for person_id, traj in data.items():
            for time, x, y in zip(traj['time'], traj['x'], traj['y']):
                f.write(f"{int(time)} {person_id} {x:.2f} {y:.2f}\n")

def main():
    input_file = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/30133_converted.txt'
    output_file_cleaned = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/refined_train/30133_converted_filtered.txt'
    output_file_interpolated = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/refined_train/30133_converted_interpolated.txt'
    output_folder = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/interpolation_plots/'

    # 读取原始数据并仅保留前50帧
    data = read_data_from_file(input_file)

    # 删除少于3帧的数据
    filtered_data = filter_short_trajectories(data)

    # 检测缺帧并删除缺帧后帧数小于等于3的行人
    clean_data = {}
    for person_id, traj in filtered_data.items():
        original_times = np.array(traj['time'])
        time_diffs = np.diff(original_times)
        missing_frames = np.sum(time_diffs > 1)
        if missing_frames > 0:
            interpolated_times, _, _ = interpolate_missing_frames(traj)
            if len(interpolated_times) > 3:  # 如果插值后帧数大于3，保留该行人数据
                clean_data[person_id] = traj

    # 保存清理后的数据
    save_data(clean_data, output_file_cleaned)

    # 对有缺帧的行人进行插值
    interpolated_data = {}
    for person_id, traj in clean_data.items():
        original_times = np.array(traj['time'])
        times, x_coords, y_coords = interpolate_missing_frames(traj)

        # 保存插值后的数据
        interpolated_data[person_id] = {'time': times, 'x': x_coords, 'y': y_coords}

        # 可视化插值点和原始点
        visualize_trajectory(person_id, times, x_coords, y_coords, original_times, output_folder)

    # 保存插值后的文件
    save_data(interpolated_data, output_file_interpolated)

    # 打印行人轨迹总数和缺帧插值的行人轨迹编号及总数
    print(f"Total number of trajectories: {len(data)}")
    print(f"Number of trajectories with missing frames and interpolated: {len(interpolated_data)}")
    print("Trajectories with missing frames and interpolated IDs:", list(interpolated_data.keys()))

if __name__ == "__main__":
    main()
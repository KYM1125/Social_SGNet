# import matplotlib.pyplot as plt

# def read_data(file_path, max_persons=10):
#     data = {}
#     with open(file_path, 'r') as file:
#         for line in file:
#             time, person_id, x, y = map(float, line.strip().split())
#             if person_id not in data:
#                 if len(data) >= max_persons:
#                     continue
#                 data[person_id] = {'time': [], 'x': [], 'y': []}
#             data[person_id]['time'].append(time)
#             data[person_id]['x'].append(x)
#             data[person_id]['y'].append(y)
#     return data

# def plot_trajectories(data):
#     plt.figure(figsize=(10, 8))
    
#     for person_id, trajectory in data.items():
#         plt.plot(trajectory['x'], trajectory['y'], marker='o', label=f'Person {int(person_id)}')
    
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')
#     plt.title('Pedestrian Trajectories (10 selected)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     # Replace 'data.txt' with the path to your data file
#     file_path = '/home/ubuntu220403/pedestrain trajetory prediction/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/pedestrians/raw/univ/test/students001.txt'
#     data = read_data(file_path, max_persons=10)
#     plot_trajectories(data)

import matplotlib.pyplot as plt
import itertools

def read_data(file_path, max_persons=10):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            time, person_id, x, y = map(float, line.strip().split())
            if person_id not in data:
                if len(data) >= max_persons:
                    continue
                data[person_id] = {'time': [], 'x': [], 'y': []}
            data[person_id]['time'].append(time)
            data[person_id]['x'].append(x)
            data[person_id]['y'].append(y)
    return data

def plot_trajectories(data, save_path=None):
    plt.figure(figsize=(10, 8))

    # 记录每个时间戳的点
    time_points = {}
    for person_id, trajectory in data.items():
        for t, x, y in zip(trajectory['time'], trajectory['x'], trajectory['y']):
            if t not in time_points:
                time_points[t] = {'x': [], 'y': []}
            time_points[t]['x'].append(x)
            time_points[t]['y'].append(y)
    
    # 绘制每个行人的轨迹
    for person_id, trajectory in data.items():
        plt.plot(trajectory['x'], trajectory['y'], marker='o', label=f'Person {int(person_id)}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Pedestrian Trajectories with Time Markers')
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.25, 1))
    plt.grid(True)
    
    # 如果指定了保存路径，保存图片
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # 使用 bbox_inches 防止图例被截断
        print(f"Image saved to {save_path}")

    plt.show()

if __name__ == "__main__":
    # Replace 'data.txt' with the path to your data file
    file_path = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/interpolated_train/30133_converted.txt'
    data = read_data(file_path, max_persons=100)
    
    # Set the save path for the image
    save_path = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots/30133_interpolated_trajectories.png'
    
    plot_trajectories(data, save_path=save_path)




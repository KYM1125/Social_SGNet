import matplotlib.pyplot as plt
import os
import numpy as np

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

def detect_missing_frames(data):
    missing_frames_ids = []
    total_persons = len(data)
    
    for person_id, trajectory in data.items():
        time_diffs = np.diff(trajectory['time'])
        if any(diff > 1 for diff in time_diffs):  # Assuming time step should be 1
            missing_frames_ids.append(person_id)
    
    return missing_frames_ids, total_persons

def plot_trajectory(person_id, trajectory, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.plot(trajectory['x'], trajectory['y'], marker='o', label=f'Person {int(person_id)}')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Pedestrian Trajectory - Person {int(person_id)}')
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Image saved to {save_path}")
    
    plt.close()

def plot_missing_frames(data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    missing_frames_ids, total_persons = detect_missing_frames(data)
    num_missing_frames = len(missing_frames_ids)

    print(f"Total number of pedestrians: {total_persons}")
    print(f"Number of pedestrians with missing frames: {num_missing_frames}")
    print(f"IDs of pedestrians with missing frames: {missing_frames_ids}")

    for person_id in missing_frames_ids:
        trajectory = data[person_id]
        save_path = os.path.join(output_folder, f'{int(person_id)}_missing_frames.png')
        plot_trajectory(person_id, trajectory, save_path=save_path)

if __name__ == "__main__":
    file_path = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/30133_converted.txt'
    output_folder = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/plots'
    
    data = read_data(file_path, max_persons=100)
    plot_missing_frames(data, output_folder)

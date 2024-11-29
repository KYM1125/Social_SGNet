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
    missing_frames_info = {}
    
    for person_id, trajectory in data.items():
        times = np.array(trajectory['time'])
        total_frames = len(times)
        
        if total_frames < 2:
            continue
        
        # Compute differences between consecutive frames
        time_diffs = np.diff(times)
        missing_frames = np.sum(time_diffs > 1)
        
        if missing_frames > 0:
            missing_frames_info[person_id] = {
                'total_frames': total_frames,
                'missing_frames': missing_frames,
                'frame_percentage': (missing_frames / total_frames) * 100
            }
    
    return missing_frames_info

def analyze_missing_frames(data, output_path):
    missing_frames_info = detect_missing_frames(data)
    total_persons = len(data)
    missing_frames_persons = {k: v for k, v in missing_frames_info.items() if v['total_frames'] <= 3}
    
    num_missing_frames_persons = len(missing_frames_info)
    num_small_trajectories = len(missing_frames_persons)

    with open(output_path, 'w') as file:
        file.write(f"Total number of pedestrians: {total_persons}\n")
        file.write(f"Number of pedestrians with missing frames: {num_missing_frames_persons}\n")
        
        file.write("\nMissing Frames Percentage for All Trajectories with Missing Frames:\n")
        for person_id, info in missing_frames_info.items():
            file.write(f"Person ID: {int(person_id)}\n")
            file.write(f"  Total Frames: {info['total_frames']}\n")
            file.write(f"  Missing Frames: {info['missing_frames']}\n")
            file.write(f"  Missing Frames Percentage: {info['frame_percentage']:.2f}%\n")
        
        file.write("\nPedestrians with total frames <= 3:\n")
        for person_id in missing_frames_persons:
            file.write(f"Person ID: {int(person_id)}\n")

if __name__ == "__main__":
    file_path = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/train/30133_converted.txt'
    output_path = '/data/user1/kym/SGNet/SGNet/SGNet/SGNet.pytorch/Trajectron-plus-plus/experiments/dair/raw/dair_infra/analysis_results.txt'

    data = read_data(file_path, max_persons=1000)
    analyze_missing_frames(data, output_path)

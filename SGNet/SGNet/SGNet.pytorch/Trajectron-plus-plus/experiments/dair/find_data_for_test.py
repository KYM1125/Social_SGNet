import os
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import shutil

def count_valid_pedestrians(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None)
    data.columns = ['timestamp', 'track_id', 'pos_x', 'pos_y']

    # 统计每个行人的帧数
    pedestrian_counts = data['track_id'].value_counts()
    
    # 统计总帧数大于50的行人数量
    valid_pedestrians = pedestrian_counts[pedestrian_counts > 50]
    return len(valid_pedestrians), os.path.basename(file_path)

def process_directory(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            results.append(count_valid_pedestrians(file_path))
    return results

def find_max_pedestrian_file(base_directory, save_directory):
    max_pedestrian_count = 0
    target_file = None
    target_file_path = ""

    directories = [os.path.join(base_directory, sub_dir) for sub_dir in ['interpolated_test', 'interpolated_val', 'interpolated_train']]
    with mp.Pool(mp.cpu_count()) as pool:
        for directory in tqdm(directories, desc="Processing directories"):
            results = pool.map(count_valid_pedestrians, [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')])
            for count, filename in results:
                if count > max_pedestrian_count:
                    max_pedestrian_count = count
                    target_file = filename
                    target_file_path = os.path.join(directory, filename)

    if target_file_path:
        # 复制文件到指定目录
        shutil.copy(target_file_path, os.path.join(save_directory, target_file))
        return target_file, max_pedestrian_count
    return None, 0

# 使用函数
base_directory = 'raw/dair_infra'  
save_directory = '.'  
os.makedirs(save_directory, exist_ok=True)  

result_file, pedestrian_count = find_max_pedestrian_file(base_directory, save_directory)

if result_file:
    print(f'The file with the most pedestrians (over 50 frames) is: {result_file} with {pedestrian_count} pedestrians. It has been copied to {save_directory}.')
else:
    print('No suitable file found.')


import os
import shutil
from tqdm import tqdm
import multiprocessing as mp

def find_target_file(base_directory, target_filename, save_directory):
    target_file_path = None

    directories = [os.path.join(base_directory, sub_dir) for sub_dir in ['all_converted']]
    for directory in tqdm(directories, desc="Processing directories"):
        file_path = os.path.join(directory, target_filename)
        if os.path.isfile(file_path):  # 检查文件是否存在
            target_file_path = file_path
            break  # 找到目标文件后退出循环

    if target_file_path:
        # 复制文件到指定目录
        shutil.copy(target_file_path, os.path.join(save_directory, target_filename))
        return target_filename
    return None

# 使用函数
base_directory = 'raw/dair_infra'
# base_directory = '/data/user1/kym/DAIR/V2X-Seq-TFD/cooperative-vehicle-infrastructure/infrastructure-trajectories'
save_directory = '.'
target_filename = '14841.txt'  # 替换为你要查找的文件名

os.makedirs(save_directory, exist_ok=True)

result_file = find_target_file(base_directory, target_filename, save_directory)

if result_file:
    print(f'The file found is: {result_file}. It has been copied to {save_directory}.')
else:
    print('No suitable file found.')

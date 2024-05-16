import os
import shutil


def move_json_files(source_dir, target_dir):
    """
    从source_dir目录中找到所有的json文件，并将它们移动到target_dir目录中。

    参数:
    - source_dir: 源目录路径，将从此目录中搜索json文件。
    - target_dir: 目标目录路径，找到的json文件将被移动到此目录。
    """

    # 确保目标目录存在，如果不存在，则创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录中的所有文件
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.json'):  # 检查文件扩展名是否为.json
            source_file_path = os.path.join(source_dir, file_name)
            target_file_path = os.path.join(target_dir, file_name)

            # 移动文件
            shutil.move(source_file_path, target_file_path)
            print(f"Moved: {file_name}")


# 使用示例
source_directory = '../data/aug_data/seg_results'  # 源目录路径，替换为实际的路径
target_directory = '../data/aug_data/seg_results_sjon'  # 目标目录路径，替换为实际的路径

# 调用函数
move_json_files(source_directory, target_directory)

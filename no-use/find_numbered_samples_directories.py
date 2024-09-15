import os
import re
import shutil
import time

def remove_numbered_samples_directories(root_dir, max_remove=1000000):
    pattern = re.compile(r'\d_samples')  # 0-9までの数字に続く'_samples'にマッチする正規表現
    removed_dirs = []
    remove_count = 0  # 削除したディレクトリの数を追跡

    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir in dirs:
            if remove_count >= max_remove:
                return removed_dirs
            if pattern.search(dir):
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_path)
                remove_count += 1
                print(f"Removed: {dir_path}")
                # time.sleep(0.1) 
    
    return removed_dirs
          

# '/path/to/directory'以下の0_samplesから9_samplesと名前のつくフォルダを最大10件削除
root_directory = './results/fashionmnist/'  # 検索を開始するディレクトリを指定
removed_directories = remove_numbered_samples_directories(root_directory)
if not removed_directories:
    print("No directories were removed.")
else:
    print(f"Total directories removed: {len(removed_directories)}")

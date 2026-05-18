import os
import shutil
import pandas as pd


source_root = '/path/to/default'      
csv_path = '/path/to/EDANSA-2019/test.csv'
dest_root = '/path/to/Edansa-test'   

# load csv
df = pd.read_csv(csv_path)

# copy files
for clip_path in df['Clip Path']:
    source_file = os.path.join(source_root, clip_path)
    dest_file = os.path.join(dest_root, clip_path)

    # Make sure the destination folder exists
    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    try:
        shutil.copy2(source_file, dest_file)
        print(f"Copied: {clip_path}")
    except FileNotFoundError:
        print(f"Missing: {clip_path}")

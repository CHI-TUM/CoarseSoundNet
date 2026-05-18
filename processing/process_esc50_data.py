import os
import pandas as pd
import shutil
from argparse import ArgumentParser

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--root",
        help="The root directory containing the ESC50 dataset master.",
        type=str
    )
    parser.add_argument(
        "--outdir",
        help="The output directory, where the filtered data shall be stored to.",
        type=str
    )
    args = parser.parse_args()
    root = args.root
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)

    mapping = {
        'crow': 'biophony',
        'insects': 'biophony',
        'chirping_birds': 'biophony',
        'crickets': 'biophony',
        'rain': 'geophony',
        'wind': 'geophony', 
        'thunderstorm': 'geophony',
        'crackling_fire': 'geophony',
        'airplane': 'anthropophony', 
        'train': 'anthropophony', 
        'helicopter': 'anthropophony', 
        'chainsaw': 'anthropophony'
    }
    # Create subdirs
    for cat in set(mapping.values()):
        os.makedirs(os.path.join(output_dir, cat), exist_ok=True)

    target_column = "category"
    df_meta = pd.read_csv(os.path.join(root, "meta/esc50.csv"))

    # filter the meta data for the mapping categories and determine the supercategories
    df_meta = df_meta[df_meta[target_column].isin(mapping.keys())]
    df_meta["supercategory"] = df_meta[target_column].apply(lambda x: mapping[x])

    # Create Bio, Geo, Anth columns based on supercategory
    df_meta['Anth'] = (df_meta['supercategory'] == 'anthropophony').astype(int)
    df_meta['Bio'] = (df_meta['supercategory'] == 'biophony').astype(int)
    df_meta['Geo'] = (df_meta['supercategory'] == 'geophony').astype(int)
    df_meta["Sil"] = 0

    # Copy the ramaining data entries to the output directory
    for row in df_meta.iterrows():
        row = row[1]
        fn = row['filename']
        src_path = os.path.join(root, "audio", fn)
        dst_path = os.path.join(output_dir, mapping[row[target_column]], fn)
        # print("SRC: ", src_path)
        # print("DST: ", dst_path)
        shutil.copy(src=src_path, dst=dst_path)
    
    # Adjust the filename column w.r.t. the supercategory
    df_meta["filename"] = df_meta.apply(lambda row: f"{row['supercategory']}/{row['filename']}", axis=1)
    print(df_meta.head())
    print(df_meta.shape)
    # Store the filtered and edited meta file to the output directory
    df_meta.to_csv(os.path.join(output_dir, "meta.csv"), index=False)

    print("Done.")
    
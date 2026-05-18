"""
Taken and adjusted from our previous repo for data mixing: https://gitlab.lrz.de/00000000014BC6CF/hts-data-mixing
"""

import argparse
import glob
import os
import pandas as pd
import librosa
import math


# Define the mapping
label_to_category = {
    'Traffic': 'Anth',
    'Airplane': 'Anth',
    'Insect': 'Bio',
    'Bird': 'Bio',
    'Rain': 'Geo',
    'Wind': 'Geo',
    'Geo': 'Geo'
}


def calculate_duration(row):
    """
    Function to calculate the duration of a file
    """
    if (row['start'] is None or row['end'] is None):
        duration = float(row['end']) - float(row['start'])
    else:
        try:
            if librosa.__version__ == '0.9.2':
                duration = librosa.get_duration(filename=row['file'])
            else:
                duration = librosa.get_duration(path=row['file'])
        except:
            duration = 0
    return duration


def split_duration(duration):
    """
    Function to split duration into segments of 10 seconds
    """
    num_segments = math.ceil(duration / TARGET_LENGTH)  # Calculate number of segments
    segments = []  # List to store segment start and end times

    # Iterate over each segment
    for i in range(num_segments):
        start_time = i * TARGET_LENGTH
        end_time = min((i + 1) * TARGET_LENGTH, duration)  
        segments.append((start_time, end_time))

    return segments


def create_segments(row):
    """
    Function to create new rows for each segment
    """
    duration = row['full_duration']
    if duration <= (TARGET_LENGTH + LENGTH_MARGIN):
        return [(None, None)]
    else:
        return split_duration(duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="The source folder containing the csvs and where to store the preprocessed csv.")
    parser.add_argument("--target_length", type=int, default=5, help="The target length of the mixture audios in seconds.")
    parser.add_argument("--margin", type=int, default=2, help="The margin which can be exceeded or undercut.")
    args = parser.parse_args()

    TARGET_LENGTH = args.target_length
    LENGTH_MARGIN = args.margin
    SRC = args.src
    DST = args.src
    # os.makedirs(DST, exist_ok=True)


    if not os.path.exists("/path/to/new_big_df.csv"):

        # Create big df with all sub csvs for preprocessing
        dfs = []
        csv_paths = glob.glob(SRC + "*.csv")
        print("Csv paths: ", csv_paths)
        if SRC + "big_df.csv" in csv_paths:
            csv_paths.remove(SRC + "big_df.csv")
        if SRC + "preprocessed_csv.csv" in csv_paths:
            csv_paths.remove(SRC + "preprocessed_csv.csv")
        # Rather take the covnerted wav files than the mp3 files from Teresa's data
        if SRC + "birds_teresa.csv" in csv_paths:
            csv_paths.remove(SRC + "birds_teresa.csv")
        if SRC + "birds_teresa_wav.csv" in csv_paths:
            csv_paths.remove(SRC + "birds_teresa_wav.csv")
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            df = df[["file", "start", "end", "split", "meta"]]
            dfs.append(df)
        
        big_df = pd.concat(dfs, ignore_index=True)

        # Check if the files actually exist
        big_df = big_df[big_df['file'].apply(os.path.exists)]
        # Apply the mapping
        df["supercategory"] = df["label"].map(label_to_category)


        print(big_df)
        big_df.to_csv(os.path.join(DST, "big_df.csv"), index=False)
    else:
        print("Big df already exist. Continue with prepocessing...")
        big_df = pd.read_csv("/path/to/new_big_df.csv")
        big_df["supercategory"] = big_df["meta"].map(label_to_category)
    

    # Use this csv for testing purposes
    # csv_path = SRC + "fsd50k.csv"
    # df = pd.read_csv(csv_path)
    df = big_df
    # First calculate the duration for every file
    df['full_duration'] = df.apply(calculate_duration, axis=1)
    # Check whether or not the file could be read and discard it if not
    df = df[df['full_duration'] != 0]
    # If the duration is longer than TARGET_LENGTH + MARGIN split the audio into segments
    df['segments'] = df.apply(lambda row: create_segments(row), axis=1)
    df = df.explode('segments')
    # Create 'start' and 'end' columns from the 'segments' tuples
    df[['start', 'end']] = pd.DataFrame(df['segments'].tolist(), index=df.index)

    # Drop the original 'segments' column
    df.drop(columns='segments', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create 'segment_id' column
    df['segment_id'] = df.groupby('file').cumcount() + 1
    print(df)

    df.to_csv(os.path.join(DST, "new_preprocessed_csv.csv"), index=False)
    print("Done.")
    







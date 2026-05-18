"""
Taken and adjusted from our previous repo for data mixing: https://gitlab.lrz.de/00000000014BC6CF/hts-data-mixing
"""

import argparse
import glob
import os
import pandas as pd
import numpy as np
import random
import librosa
import audtorch.transforms as T
import audtorch.transforms.functional as F
import math
import audiofile
import json
import yaml
import auglib
import torch
from tqdm import tqdm

def load_audio(row, sample_rate, target_length):
    try:
        fp = row["file"]
        start = float(row["start"])
        end = float(row["end"])
        # Apply a GainStage for each audio that is loaded
        gain_db = random.sample(range(-30, 1), 1)[0]
        gainstage = auglib.transform.GainStage(gain_db=gain_db)
        # Load the audio, either from start to end or the complete audio
        if not (math.isnan(start) or math.isnan(end)):
            audio, sr = librosa.load(fp, sr=sample_rate, offset=float(start), duration=end-start)
        else:
            audio, sr = librosa.load(fp, sr=sample_rate)
        # RandomCrop or Expand the audio to the target_length
        # RandomCrop apparently also expands the audio if it is too short with the method specified
        audio = T.RandomCrop(int(target_length * sample_rate), method='replicate')(audio)
        audio = gainstage(audio, sampling_rate=sample_rate)
        return audio, sr, gain_db
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None, None


def mix_audios(files_to_mix: pd.DataFrame) -> (torch.tensor, list, list, list):
    snrs = random.sample(range(-5, 6), len(files_to_mix) - 1)
    gain_dbs = []
    unused_files = []

    # If there are at least two audio files to mix, mix them
    if len(files_to_mix) > 1:
        mixed_audio = None
        for i, row in enumerate(files_to_mix.iterrows()):
            row = row[1]
            new_audio, sr, gain_db = load_audio(row, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)
            if new_audio is None:
                unused_files.append(row["file"])
                continue
            gain_dbs.append(gain_db)
            # audiofile.write(f"./mixings/{row['meta']}_{gain_db}_{random.randint(0, 100000000)}.wav", new_audio, sampling_rate=SAMPLE_RATE)
            if mixed_audio is None:
                mixed_audio = new_audio
            else:
                mixed_audio = F.additive_mix(mixed_audio, new_audio, snrs[i-1])
                mixed_audio = T.Normalize()(mixed_audio)
    # Otherwise just take the one audio
    else:
        mixed_audio, sr, gain_db = load_audio(files_to_mix.iloc[0], sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH)
        gain_dbs.append(gain_db)
        # audiofile.write(f"./mixings/{row['meta']}_{gain_db}_{random.randint(0, 100000000)}.wav", mixed_audio, sampling_rate=SAMPLE_RATE)

    return mixed_audio, snrs, gain_dbs, unused_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="The source folder containing the preprocessed csv.")
    parser.add_argument("--dst", type=str, help="The destination folder where to store the mixed files.")
    parser.add_argument("--split", type=str, default="train", help="Specify whether the train or test mixtures shall be generated.")
    parser.add_argument("--sr", type=int, default=32000, help="The sampling rate that shall be used for the mixtures.")
    # parser.add_argument("--max_samples", type=int, default=5, help="Tha maximum number of source files that shall be mixed.")
    parser.add_argument("--target_length", type=int, default=5, help="The target length of the mixture audios in seconds.")
    # parser.add_argument("--margin", type=int, default=2, help="The margin which can be exceeded or undercut.")
    args = parser.parse_args()

    TARGET_LENGTH = args.target_length
    # LENGTH_MARGIN = args.margin
    SAMPLE_RATE = args.sr
    # MAX_SAMPLE_NUMBER = args.max_samples
    SPLIT_NAME = args.split
    SRC = args.src
    DST = args.dst
    os.makedirs(DST, exist_ok=True)

    noise_functions = {
        "WNG": auglib.transform.WhiteNoiseGaussian, 
        "WNU": auglib.transform.WhiteNoiseUniform, 
        "PIN": auglib.transform.PinkNoise
    }

    random.seed(1)
    # df = pd.read_csv(SRC + "preprocessed_csv_wavBirds.csv")
    df = pd.read_csv(SRC + "new_big_df.csv")
    # Separate the DataFrame into train and test splits
    df = df[df['split'] == SPLIT_NAME].reset_index(drop=True)
    # test_df = df[df['split'] == 'test']

    df_A = df[df["supercategory"] == "Anth"]
    df_B = df[df["supercategory"] == "Bio"]
    df_G = df[df["supercategory"] == "Geo"]

    target_combinations = {
        "A": 8500,
        "B": 8500,
        "G": 8500,
        "AB": 8500,
        "AG": 8500,
        "BG": 8500,
        "ABG": 8500,
    }

    # # Define the custom probabilities for the number of samples
    # choices = [1, 2, 3]
    # weights = [5, 3, 1]

    for combo, num_mixes in target_combinations.items():
        print("Combination: ", combo)
        dict_list = []

        # Adjust choices and weights based on the number of different categories
        if len(combo) == 1:
            choices = [1, 2, 3, 4]
            weights = [1, 5, 5, 1]
        if len(combo) == 2:
            choices = [1, 2, 3]
            weights = [5, 3, 1]
        else:
            choices = [1, 2]
            weights = [5, 1]

        # Generate the mixtures for the current combo
        for id_counter in tqdm(range(num_mixes)):
            files_to_mix = []

            # Select how many files from a category shall be taken
            if "A" in combo:
                num_from_A = random.choices(choices, weights=weights)[0]
                files_from_A = df_A.sample(num_from_A)
                files_to_mix.append(files_from_A)

            if "B" in combo:
                num_from_B = random.choices(choices, weights=weights)[0]
                files_from_B = df_B.sample(num_from_B)
                files_to_mix.append(files_from_B)

            if "G" in combo:
                num_from_G = random.choices(choices, weights=weights)[0]
                files_from_G = df_G.sample(num_from_G)
                files_to_mix.append(files_from_G)
            
            # Convert the list of sub dfs into one df
            files_to_mix = pd.concat(files_to_mix, ignore_index=True)
            # print("Files to mix:\n", files_to_mix)
            mixed_audio, snrs, gain_dbs, unused_files = mix_audios(files_to_mix=files_to_mix)

            # If none of the files could be loaded remove them from the source dataframes and continue
            if mixed_audio is None:
                df_A = df_A[~df_A['file'].isin(files_to_mix['file'])]
                df_B = df_B[~df_B['file'].isin(files_to_mix['file'])]
                df_G = df_G[~df_G['file'].isin(files_to_mix['file'])]
                continue

            # Remove unused files from the dataframes (apparently errors during loading the audio)
            if len(unused_files) > 0:
                df_A = df_A[~df_A['file'].isin(unused_files)]
                df_B = df_B[~df_B['file'].isin(unused_files)]
                df_G = df_G[~df_G['file'].isin(unused_files)]
                continue

            # Add noise to the mixture with a probability of 0.5 
            add_noise = random.random() >= 0.5
            noise_key = "None"
            snr_db = None
            if add_noise:
                noise_key, noise_function = random.choice(list(noise_functions.items()))
                snr_db = random.sample(range(-5, 16), 1)[0]
                noise_function = noise_function(snr_db=snr_db)
                mixed_audio = noise_function(mixed_audio)
            
            # Store the mixed audio
            os.makedirs(os.path.join(DST, SPLIT_NAME, combo), exist_ok=True)
            audiofile.write(os.path.join(DST, SPLIT_NAME, combo, "{:012d}.wav".format(id_counter)), mixed_audio, sampling_rate=SAMPLE_RATE)

            # remove unused files (due to errors or the like) from the randomly sampled files
            files_to_mix = files_to_mix[~files_to_mix['file'].isin(unused_files)]
            # create meta info of the current mixing iteration
            mix_dict = {
                # "mix_id": id_counter,
                "mix_id": "{:012d}".format(id_counter),
                "mix_files": len(files_to_mix),
                "files": files_to_mix['file'].tolist(), # files will be mixed in the order with which they appear in the list, so first f0 with f1, then this result with f2, then that result with f3, and so on...
                "gain_dbs": gain_dbs,
                "snrs": snrs, # we ony need mix_files - 1 SNRs, as they define
                "labels": sorted(list(set(files_to_mix['supercategory']))),
                "noise": noise_key,
                "noise_snr_db": snr_db if not noise_key == "None" else "None"
            }
            # print("\n", mix_dict)
            dict_list.append(mix_dict)

            # Remove files used for selection from DataFrame and increase counter -> for the new sampling with predefined samples we don't do that
            # df = df[~df['file'].isin(files_to_select['file'])]

        # Store the mixing information as json and yaml files
        with open(os.path.join(DST, SPLIT_NAME, combo, f"mix_info_{SPLIT_NAME}.json"), "w") as f:
            json.dump(dict_list, f, indent=4)
        
        with open(os.path.join(DST, SPLIT_NAME, combo, f"mix_info_{SPLIT_NAME}.yaml"), 'w') as yaml_file:
            yaml.dump(dict_list, yaml_file, default_flow_style=False)

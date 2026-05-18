"""
Taken and adjusted from our previous repo for data mixing: https://gitlab.lrz.de/00000000014BC6CF/hts-data-mixing
"""

import argparse
import os
import pandas as pd
import numpy as np
import random
import audiofile
import json
import yaml
import auglib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dst", type=str, help="The destination folder where to store the created silence audios.")
    parser.add_argument("--target_length", type=int, default=5, help="The target length of the silence audios in seconds.")
    parser.add_argument("--sr", type=int, default=32000, help="The sampling rate for the silence audios.")
    parser.add_argument("--num", type=int, default=10000, help="The number of silence audios that shall be created.")
    args = parser.parse_args()

    TARGET_LENGTH = args.target_length
    SAMPLE_RATE = args.sr
    NUMBER_TO_GENERATE = args.num
    DST = "/path/to/synthetic_data/silence"
    os.makedirs(DST, exist_ok=True)

    noise_functions = {
        "WNG": auglib.transform.WhiteNoiseGaussian, 
        "WNU": auglib.transform.WhiteNoiseUniform, 
        "PIN": auglib.transform.PinkNoise
    }

    dict_list = []
    for i in range(NUMBER_TO_GENERATE):
        audio = np.zeros(shape=(SAMPLE_RATE * TARGET_LENGTH,))

        # Add noise
        noise_key, noise_function = random.choice(list(noise_functions.items()))
        gain_db_initial = random.sample(range(-5, 2), 1)[0]
        noise_function = noise_function(gain_db=gain_db_initial)
        audio = noise_function(audio)

        # Apply GainStage from -40 to -5
        gain_db = random.sample(range(-40, -4), 1)[0]
        gainstage = auglib.transform.GainStage(gain_db=gain_db)
        audio = gainstage(audio, sampling_rate=SAMPLE_RATE)

        # Save the audio file
        audiofile.write(os.path.join(DST, "{:012d}.wav".format(i)), audio, sampling_rate=SAMPLE_RATE)

        silence_dict = {
            "silence_id": "{:012d}".format(i),
            "noise": noise_key,
            "gain_db_initial": gain_db_initial,
            "gain_db": gain_db
        }
        dict_list.append(silence_dict)
    
    # Store the mixing information as json and yaml files
    with open(os.path.join(DST, f"silence_info.json"), "w") as f:
        json.dump(dict_list, f, indent=4)
    
    with open(os.path.join(DST, f"silence_info.yaml"), 'w') as yaml_file:
        yaml.dump(dict_list, yaml_file, default_flow_style=False)
    print("Done.")

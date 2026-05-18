import argparse
import os
import pandas as pd
import random
import librosa
import audtorch.transforms as T
import audtorch.transforms.functional as F
import audiofile
import json
import yaml
import auglib

def load_audio(row, sample_rate, target_length, dirpath):
    try:
        fp = os.path.join(dirpath, row["filename"])
        gain_db = random.sample(range(-30, 1), 1)[0]
        gainstage = auglib.transform.GainStage(gain_db=gain_db)
        audio, sr = librosa.load(fp, sr=sample_rate)
        audio = T.RandomCrop(int(target_length * sample_rate), method='replicate')(audio)
        audio = gainstage(audio, sampling_rate=sample_rate)
        return audio, sr, gain_db
    except Exception as e:
        print(f"Error loading audio: {str(e)}")
        return None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="Filepath of the esc50 meta.csv file.")
    parser.add_argument("--dst", type=str, help="Destination folder for mixed files.")
    parser.add_argument("--sr", type=int, default=44100, help="Sampling rate.")
    parser.add_argument("--target_length", type=int, default=5, help="Target length in seconds.")
    args = parser.parse_args()
    
    SAMPLE_RATE = args.sr
    TARGET_LENGTH = args.target_length
    SRC = args.src
    DST = args.dst
    os.makedirs(DST, exist_ok=True)
    
    noise_functions = {
        "WNG": auglib.transform.WhiteNoiseGaussian, 
        "WNU": auglib.transform.WhiteNoiseUniform, 
        "PIN": auglib.transform.PinkNoise
    }
    
    random.seed(1)
    df = pd.read_csv(SRC)
    grouped = df.groupby("supercategory")
    supercategories = ["biophony", "geophony", "anthropophony"]
    
    dict_list = []
    id_counter = 0
    
    while not df.empty:
        print(f"\nGENERATE MIXING NUMBER {id_counter}")
        
        # Randomly determine how many sources to mix (1, 2, or 3)
        num_sources = random.choices([1, 2, 3], weights=[3, 4, 1])[0]
        
        available_categories = [cat for cat in supercategories if cat in grouped.groups]
        selected_categories = random.sample(available_categories, min(num_sources, len(available_categories)))
        
        selected_files = []
        for cat in selected_categories:
            sample = grouped.get_group(cat).sample(1)
            selected_files.append(sample.iloc[0])
        
        snrs = random.sample(range(-5, 6), max(0, len(selected_files) - 1))
        gain_dbs = []
        mixed_audio = None
        
        for i, row in enumerate(selected_files):
            new_audio, sr, gain_db = load_audio(row, sample_rate=SAMPLE_RATE, target_length=TARGET_LENGTH, dirpath=os.path.dirname(SRC))
            if new_audio is None:
                continue
            gain_dbs.append(gain_db)
            if mixed_audio is None:
                mixed_audio = new_audio
            else:
                mixed_audio = F.additive_mix(mixed_audio, new_audio, snrs[i-1])
                mixed_audio = T.Normalize()(mixed_audio)
        
        if mixed_audio is None:
            continue
        
        add_noise = random.random() >= 0.5
        noise_key = "None"
        snr_db = None
        if add_noise:
            noise_key, noise_function = random.choice(list(noise_functions.items()))
            snr_db = random.sample(range(-5, 16), 1)[0]
            noise_function = noise_function(snr_db=snr_db)
            mixed_audio = noise_function(mixed_audio)
        
        os.makedirs(os.path.join(DST, "mixings_wav"), exist_ok=True)
        audiofile.write(os.path.join(DST, "mixings_wav", "{:012d}.wav".format(id_counter)), mixed_audio, sampling_rate=SAMPLE_RATE)
        
        mix_dict = {
            "mix_id": "{:012d}".format(id_counter),
            "files": [row["filename"] for row in selected_files],
            "gain_dbs": gain_dbs,
            "snrs": snrs,
            "labels": [row["category"] for row in selected_files],
            "supercategories": [row["supercategory"] for row in selected_files],
            "noise": noise_key,
            "noise_snr_db": snr_db if not noise_key == "None" else "None"
        }
        dict_list.append(mix_dict)
        
        df = df[~df['filename'].isin([row["filename"] for row in selected_files])]
        grouped = df.groupby("supercategory")
        id_counter += 1
    
    with open(os.path.join(DST, "mix_info.json"), "w") as f:
        json.dump(dict_list, f, indent=4)
    
    with open(os.path.join(DST, "mix_info.yaml"), 'w') as yaml_file:
        yaml.dump(dict_list, yaml_file, default_flow_style=False)

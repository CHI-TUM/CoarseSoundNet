import os
import pandas as pd
import librosa
from pydub import AudioSegment
from tqdm import tqdm

if __name__=='__main__':
    path = "/path/to/coarse/EDANSA-2019"
    beambient = "/path/to/Svenja_data/segments_5s"
    htsforest = "/path/to/coarse/Dominik_additional_data"
    akwamo = "/path/to/akwamo_coarse_paper"
    publicsynth = "/path/to/synthetic_data"
    besound = "/path/to/coarse/BE_data/original"


folders = {
    "beambient": "/path/to/Svenja_data/segments_5s",
    "htsforest": "/path/to/Dominik_additional_data",
    "akwamo": "/path/to/akwamo_coarse_paper",
    "publicsynth": "/path/to/synthetic_data"    
}

for key, path in folders.items():
    print(f"\n--- {key} ---")
    train, dev, test = pd.read_csv(os.path.join(path, "train.csv")), pd.read_csv(os.path.join(path, "dev.csv")), pd.read_csv(os.path.join(path, "test.csv"))
    df = pd.concat([train, dev, test], axis=0)

    total_duration = 0.0
    for fn in tqdm(df["filename"].values):
        fp = os.path.join(path, "default", fn)
        audio, sr = librosa.load(fp, sr=None)
        duration = librosa.get_duration(y=audio, sr=sr)
        total_duration += duration
    print(f"Total duration {key}: {total_duration:.2f} seconds")
    print(f"Total duration {key}: {total_duration/3600:.2f} hours")



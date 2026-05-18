import os
import pandas as pd
from pydub import AudioSegment

root_dir = "/path/to/data"

for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(dirpath, filename)
            wav_path = mp3_path[:-4] + ".wav"
            
            try:
                sound = AudioSegment.from_mp3(mp3_path)
                # Don’t set frame_rate → keep original sampling rate
                sound.export(wav_path, format="wav")
                print(f"Converted: {mp3_path} → {wav_path}")
            except Exception as e:
                print(f"Error converting {mp3_path}: {e}")




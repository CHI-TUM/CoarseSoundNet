import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def split_audio(file_path, max_length=10):
    """Splits an audio file into 10-second segments"""
    audio, sample_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sample_rate)

    if duration <= max_length:
        return [(audio, sample_rate, 0)]  # No need to split if <= 10 sec

    parts = []
    start_time = 0

    while start_time < duration:
        end_time = start_time + max_length
        if end_time > duration:
            end_time = duration  # Avoid exceeding file duration
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        part = audio[start_sample:end_sample]

        parts.append((part, sample_rate, start_time))
        start_time += max_length  # Move to next segment

    return parts


def process_audio_files(input_directory, output_directory, annotations_csv, output_csv):
    """Processes audio files and creates a new annotation CSV with 10-second segments"""
    
    # Load CSV
    df = pd.read_csv(annotations_csv)

    new_annotations = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                parts = split_audio(file_path)

                relative_path = os.path.relpath(root, input_directory)
                output_subdirectory = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdirectory, exist_ok=True)

                for i, (part, sample_rate, segment_start) in tqdm(enumerate(parts)):
                    new_filename = f"{os.path.splitext(file)[0]}_segment{i+1}.wav"
                    new_file_path = os.path.join(output_subdirectory, new_filename)

                    sf.write(new_file_path, part, sample_rate)
                    print(f"Saved {new_file_path}")

                    # Update CSV annotations
                    matching_rows = df[df['filename'] == os.path.splitext(file)[0]]

                    for _, row in matching_rows.iterrows():
                        new_start = max(0, row['beginn_time'] - segment_start)
                        new_end = max(0, row['end_time'] - segment_start)

                        if new_start < 10 and new_end > 0:  # If annotation falls within this 10s segment
                            new_annotations.append({
                                "time": row["time"],
                                "date": row["date"],
                                "RecorderID": row["RecorderID"],
                                "filename": new_filename,
                                "beginn_time": max(0, new_start),  # Keep within bounds
                                "end_time": min(10, new_end),  # Ensure it fits 10s segment
                                "low_freq": row["low_freq"],
                                "high_freq": row["high_freq"],
                                "Code": row["Code"],
                                "duration": row["duration"],
                                "freq_range": row["freq_range"],
                                "Soundgroup": row["Soundgroup"],
                                "Phony_Class": row["Phony_Class"]
                            })

    # Save new CSV
    new_df = pd.DataFrame(new_annotations)
    new_df.to_csv(output_csv, index=False)
    print(f"Saved updated annotations to {output_csv}")

if __name__ == '__main__':
    input_directory = '/path/to/input/dir'
    output_directory = '/path/to/output/Britz_segments'
    annotations_csv = '/path/to/Annotations_Britz.csv'
    output_csv = '/path/to/Annotations_Britz_segments.csv'

    process_audio_files(input_directory, output_directory, annotations_csv, output_csv)

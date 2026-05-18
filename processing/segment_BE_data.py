import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def split_audio(file_path, max_length=10):
    """Splits an audio file into max_length-second segments"""
    audio, sample_rate = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=audio, sr=sample_rate)

    if duration <= max_length:
        return [(audio, sample_rate, 0)]  # No need to split if <= max_length sec

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


def process_audio_files(input_directory, output_directory, annotations_csv, output_csv, max_length=10):
    """Processes audio files and creates a new annotation CSV with max_length-second segments"""
    
    # Load CSV
    df = pd.read_csv(annotations_csv)

    new_annotations = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav')):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")

                parts = split_audio(file_path, max_length=max_length)

                relative_path = os.path.relpath(root, input_directory)
                output_subdirectory = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdirectory, exist_ok=True)

                for i, (part, sample_rate, segment_start) in tqdm(enumerate(parts)):
                    new_filename = f"{os.path.splitext(file)[0]}_segment{i+1}.wav"
                    new_file_path = os.path.join(output_subdirectory, new_filename)

                    sf.write(new_file_path, part, sample_rate)
                    print(f"Saved {new_file_path}")

                    # Get the filename with the relative path and the entries in the dataframe
                    relative_filename = os.path.join(relative_path, file)
                    matching_rows = df[df['file'] == relative_filename]

                    # Update CSV annotations
                    # matching_rows = df[df['file'] == os.path.splitext(file)[0]]

                    for _, row in matching_rows.iterrows():
                        new_start = max(0, row['Begin Time (s)'] - segment_start)
                        new_end = max(0, row['End Time (s)'] - segment_start)

                        if new_start < max_length and new_end > 0:  # If annotation falls within this max_length s segment
                            new_annotations.append({
                                "file": os.path.join(relative_path, new_filename),
                                "Selection": row["Selection"],
                                "View": row["View"],
                                "Channel": row["Channel"],
                                "Begin Time (s)": max(0, new_start),
                                "End Time (s)": min(max_length, new_end),
                                "Low Freq (Hz)": row["Low Freq (Hz)"],
                                "High Freq (Hz)": row["High Freq (Hz)"],
                                "Delta Time (s)": row["Duration"],
                                "Delta Freq (Hz)": row["Delta Freq (Hz)"],
                                "Avg Power Density (dB FS/Hz)": row["Avg Power Density (dB FS/Hz)"],
                                "Annotation": row["Annotation"],
                                "Plot": row["Plot"],
                                "Date": row["Date"],
                                "Duration": row["Duration"],
                                "Superclass": row["Superclass"],
                                "Hours": row["Hours"]
                            })

    # Save new CSV
    new_df = pd.DataFrame(new_annotations)
    new_df.to_csv(output_csv, index=False)
    print(f"Saved updated annotations to {output_csv}")

if __name__ == '__main__':
    input_directory = '/path/to/BE_data/original'
    output_directory = '/path/to/BE_data/segments'
    annotations_csv = '/path/to/annotations_BEForest.csv'
    output_csv = '/path/to/annotations_BEForest_segments.csv'
    MAX_LENGTH = 5 # 10 

    process_audio_files(input_directory, output_directory, annotations_csv, output_csv, max_length=MAX_LENGTH)

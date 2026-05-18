import os
import time
import yaml
import numpy as np
from autrainer.serving import Inference

model_paths = {
    'Cnn10': "path/to/best/model",
    # ...
}

file = "/path/to/BE_data/original/0530-31032016_304.wav" 
print("Analysis file: ", file)
N_RUNS = 1000
print("Number of runs: ", N_RUNS)

for model, model_path in model_paths.items():
    print()
    print("Model: ", model)
    
    # Get the sample rate
    yaml_path = os.path.join(model_path, "preprocess_file_handler.yaml")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    key = next(iter(data))
    sample_rate = data[key]['target_sample_rate']
    print("Sample rate: ", sample_rate)

    # Initialize the inference class
    inference = Inference(
        model_path = model_path,
        device = "cuda:0",
        window_length = 10,
        stride_length = 10,
        sample_rate = sample_rate,
    )

    # Pre-load the audio file
    x = inference.file_handler.load(file)

    # Compute the mean inference time for a model
    times = []
    for i in range(N_RUNS):
        start_time = time.time()

        if inference._window_length and inference._stride_length and inference._sample_rate:
            pred = inference._predict_windowed(x)
        else:
            pred = inference._predict(x)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    mean_time = np.mean(times)
    print("Mean inference time: ", mean_time)

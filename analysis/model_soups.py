import os
import pandas as pd
import numpy as np
import torch
from collections import OrderedDict
from glob import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_category = "w2v2"
model = "w2v2-l"
search_string = f"/path/to/different_models_woSilence/{model_category}/training/CoarseNet-32k-ML-Edansa-{model_category}_{model}_*/_best/model.pt"
print(search_string)

model_paths = glob(search_string)
print(model_paths)
print(len(model_paths))
if len(model_paths) == 0:
    print("No model paths were found. Exit.")
    exit()

# Load the weights for every setting
state_dicts = [torch.load(path, map_location=device) for path in model_paths]
# print(state_dicts)
print(len(state_dicts))
print(len(state_dicts[0]))


# Uniform soup: averaging the weights
avg_state_dict = OrderedDict()
for key in state_dicts[0].keys():
    avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)

print(len(avg_state_dict))

for mp in model_paths:
    print("Model path: ", mp)
    out_path = os.path.dirname(os.path.dirname(mp))
    out_path = os.path.join(out_path, "_model_soups_uniform", "model.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print("Saving to: ")
    print(os.path.dirname(out_path))
    print("as")
    print(out_path)
    
    torch.save(avg_state_dict, out_path)

print("Saved uniform model soup.")

# Possible TODO in the future:
# Greedy soup: only taking weights into account, when they improve the model performance on the validation set    




import os
import pandas as pd
import numpy as np
import torch
import autrainer.cli
from collections import OrderedDict
from glob import glob
from sklearn.metrics import classification_report, f1_score


model_paths = {
    'Cnn10': "/path/to/best/model/state",
    # ...
}

test_csv_path = '/path/to/EDANSA-2019/test.csv'
checkpoint_name = "_model_soups_uniform"
input_dir = "/path/to/coarse/Edansa-test"
target_names = ["Anth", "Bio", "Geo"]
columns_to_keep = ["filename", "Anth", "Bio", "Geo"]

THRESHOLD = 0.5
predict = False

if predict:
    print("Inference:")
    for model, mp in model_paths.items():
        print("\nModel: ", model)
        output_dir = os.path.join(mp, checkpoint_name)

        autrainer.cli.inference(
            model=mp,
            input=input_dir,
            output=output_dir,
            checkpoint=checkpoint_name,
            device="cuda:0",
            recursive=True
        )
else:
    print("Evaluation:")
    for model, mp in model_paths.items():
        print("\nModel: ", model)

        y_pred_probs = pd.read_csv(os.path.join(mp, checkpoint_name, "results.csv"))
        y_pred_probs = y_pred_probs[columns_to_keep]
        y_pred_probs.set_index("filename", inplace=True)

        y_true = pd.read_csv(test_csv_path)
        y_true = y_true.rename(columns={"Clip Path": "filename"})
        y_true = y_true[columns_to_keep]
        y_true.set_index("filename", inplace=True)

        # Align the two dataframes
        y_true = y_true.loc[y_pred_probs.index]
        y_pred_probs = y_pred_probs.loc[y_true.index]

        # Convert predictions to binary and dataframes to array
        y_pred_probs = y_pred_probs.values
        y_pred_bin = (y_pred_probs > THRESHOLD).astype(float)
        y_true = y_true.values

        # Final check
        # print(y_pred_probs[0], y_pred_probs.shape, y_pred_probs.dtype)
        # print(y_pred_bin[0], y_pred_bin.shape, y_pred_bin.dtype)
        # print()
        # print(y_true[0], y_true.shape, y_true.dtype)

        print(classification_report(y_true, y_pred_bin, target_names=target_names, zero_division=0))
        print()
        print("F1-Scores:")
        print("Macro: ", f1_score(y_true, y_pred_bin, average="macro"))
        print("Weighted: ", f1_score(y_true, y_pred_bin, average="weighted"))
        print("Class-wise: ", f1_score(y_true, y_pred_bin, average=None))
        print("Anth: ", f1_score(y_true[:, 0], y_pred_bin[:, 0], average="binary"))
        print("Bio: ", f1_score(y_true[:, 1], y_pred_bin[:, 1], average="binary"))
        print("Geo: ", f1_score(y_true[:, 2], y_pred_bin[:, 2], average="binary"))
        print("\n----------------")
        
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# target_columns = ["Anth", "Bio", "Geo", "Sil"]
target_columns = ["Anth", "Bio", "Geo"]
thresholds = {
    "Anth": 0.5,
    "Bio": 0.5,
    "Geo": 0.5,
    "Sil": 0.5
}

model_path = "/path/to/trained/model"

best_f1 = 0.0
best_epoch = 0
average = "weighted"
for epoch in range(1, 31):
    print(f"\nEPOCH {epoch}")
    current_state = os.path.join(model_path, f"epoch_{epoch}")
    # print(current_state)

    dev_targets = np.load(os.path.join(current_state, "dev_targets.npy"))#[:, :-1]
    # print(dev_targets.shape, dev_targets[0])
    # print(dev_targets.shape, dev_targets[0], dev_targets.dtype)

    dev_results = pd.read_csv(os.path.join(current_state, "dev_results.csv"))
    dev_results = dev_results[target_columns]
    preds = dev_results.apply(lambda col: (col > thresholds[col.name]).astype(float))
    # print(preds.shape, preds.head())

    current_f1 = f1_score(dev_targets, preds, average=average)
    print(f"F1 {average}: ", current_f1)

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_epoch = epoch

print("\nBest epoch: ", best_epoch)
print("F1-score: ", best_f1)
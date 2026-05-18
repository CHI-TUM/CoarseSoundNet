import json
import pandas as pd

json_path = "/path/to/mix_info.json"
# Open and load the JSON file
with open(json_path, "r") as file:
    json_data = json.load(file)

# Convert JSON to DataFrame
df = pd.DataFrame(json_data)

# Create columns for each category and set values based on 'supercategories'
categories = {"Anth": "anthropophony", "Bio": "biophony", "Geo": "geophony", "Sil": "silence"}
for col, category in categories.items():
    df[col] = df["supercategories"].apply(lambda x: 1 if category in x else 0)

print(df.head())
# Select only relevant columns for output
df = df[["mix_id", "Anth", "Bio", "Geo", "Sil"]]

# Display the result
print(df)

print("\nLabel distribution across categories:")
print(df[categories.keys()].sum(0))
print("\nNumber of used sources for the mixtures:")
print(df[categories.keys()].sum(1).value_counts())

df.to_csv(json_path.replace(".json", ".csv"), index=False)
print("Saved csv file.")
import pandas as pd

label_to_category = {
    'Traffic': 'Anth',
    'Airplane': 'Anth',
    'Insect': 'Bio',
    'Bird': 'Bio',
    'Rain': 'Geo',
    'Wind': 'Geo',
    'Geo': 'Geo'
}


df_pp = pd.read_csv("/path/to/new_big_df.csv")
print(df_pp.head(), df_pp.shape)
print(df_pp["meta"].value_counts())

df_pp["supercategory"] = df_pp["meta"].map(label_to_category)
print(df_pp.head(), df_pp.shape)
print(df_pp["supercategory"].value_counts())
df_pp.to_csv("/path/to/new_big_df.csv", index=False)

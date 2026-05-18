"""
Convert single-labelled csv files to one-hot encoded csv files for multi-label classification.
"""

import os
import pandas as pd

if __name__=='__main__':
    root = "/path/to/folder"
    
    target_column = 'label'
    splits = ["df_train.csv", "df_dev.csv", "df_test.csv"]
    THRESHOLD = 50
    
    for split in splits:
        df = pd.read_csv(os.path.join(root, split))
        df.rename(columns={'segment_filepath': 'filename'}, inplace=True)
        # Added the following line for the david data
        df['filename'] = df.apply(lambda x: f"{x['code_unique']}_segment_{x['chunk_initial_time']}_{x['chunk_final_time']}.wav", axis=1)
        # one_hot_encoded_df = pd.get_dummies(df, columns=["species"])

        multilabel_df = df.groupby('filename')[target_column].apply(lambda x: x.unique()).reset_index()
        multilabel_df = multilabel_df.explode(target_column)
        # Create a binary indicator for each annotation
        multilabel_df = multilabel_df.pivot_table(index='filename', columns=target_column, aggfunc='size', fill_value=0)
        # Convert the DataFrame to binary (1 or 0)
        multilabel_df[multilabel_df > 0] = 1
        multilabel_df.reset_index(inplace=True)
        print(multilabel_df.head(5))
        print(multilabel_df.shape)

        # Only keep the species columns that sum up to over the predefined threshold
        col_keep = pd.Series(multilabel_df.iloc[:, 1:].sum(axis=0) > THRESHOLD)
        col_keep['filename'] = True
        multilabel_df = multilabel_df.loc[:, col_keep[col_keep].index]
        columns = ['filename'] + [col for col in multilabel_df.columns if col != 'filename']
        multilabel_df = multilabel_df[columns]
        print("\nNew:")
        print(multilabel_df.head(5))
        print(multilabel_df.shape)


        multilabel_df.to_csv(os.path.join(root, split.replace(".csv", "_one_hot.csv")), index=False)

    print("Species:\n", list(multilabel_df.columns[1:]))
    print("Num species:\n", len(multilabel_df.columns[1:]))
    print("Num samples without active label: ", sum(multilabel_df.iloc[:, 1:].sum(axis=1) == 0))

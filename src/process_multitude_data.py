"""
Parses the MULTITuDE csv into json objects that match the format of the data
expected by the text fluoroscopy pipeline.
"""

import pandas as pd
import torch
import json
import os
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 150
VALID_SIZE = 150
TEST_SIZE = 200
SEED = 42

# Open MULTITuDE csv
output_dir = 'dataset/processed_multitude'
csv_file = 'dataset/raw/multitude.csv'
data = pd.read_csv(csv_file)

# Rename column 'label' to 'result'
data.rename(columns={'label': 'result'}, inplace=True)

# Make directories
os.makedirs(f"{output_dir}/labels", exist_ok=True)
os.makedirs(f"{output_dir}/test_data", exist_ok=True)
os.makedirs(f"{output_dir}/train_valid_data", exist_ok=True)

# Combine all their data and re-split it
# Group by language
grouped_by_language = data.groupby('language')

for language, lang_df in grouped_by_language:
    print("Processing", language)
    # # First get human samples for this language
    human_df = lang_df[lang_df['result'] == 0]
    llm_df = lang_df[lang_df['result'] == 1]

    # Use some of the data for a separate general test split
    human_df, test_hdf = train_test_split(
        human_df, 
        test_size=TEST_SIZE // 2, 
        random_state=SEED,
        shuffle=True
    )
    llm_df, test_ldf = train_test_split(
        llm_df, 
        test_size=TEST_SIZE // 2, 
        random_state=SEED,
        shuffle=True
    )
    # Use some of the data for a valid split
    human_df, valid_hdf = train_test_split(
        human_df, 
        test_size=VALID_SIZE // 2, 
        random_state=SEED, 
        shuffle=True
    )
    llm_df, valid_ldf = train_test_split(
        llm_df, 
        test_size=VALID_SIZE // 2, 
        random_state=SEED,
        shuffle=True
    )
    # Use some of the data for a train split
    human_df, train_hdf = train_test_split(
        human_df, 
        test_size=TRAIN_SIZE // 2, 
        random_state=SEED, 
        shuffle=True
    )
    llm_df, train_ldf = train_test_split(
        llm_df, 
        test_size=TRAIN_SIZE // 2, 
        random_state=SEED, 
        shuffle=True
    )

    # Format as list of dicts
    test_df = pd.concat([test_hdf, test_ldf]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    valid_df = pd.concat([valid_hdf, valid_ldf]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    train_df = pd.concat([train_hdf, train_ldf]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    test_data = test_df[["text", "result"]].to_dict(orient="records")  
    valid_data = valid_df[["text", "result"]].to_dict(orient="records")  
    train_data = train_df[["text", "result"]].to_dict(orient="records")  

    # Extract just the labels and save as torch tensors
    test_labels = torch.tensor(test_df["result"].astype(int).tolist())
    valid_labels = torch.tensor(valid_df["result"].astype(int).tolist())
    train_labels = torch.tensor(train_df["result"].astype(int).tolist())

    torch.save(train_labels, f"{output_dir}/labels/{language}_train.pt")
    torch.save(valid_labels, f"{output_dir}/labels/{language}_val.pt")
    torch.save(test_labels, f"{output_dir}/labels/{language}_test.pt")

    with open(f"{output_dir}/train_valid_data/{language}_train.json", 'w', encoding='utf-8') as output_file:
        json.dump(train_data, output_file, ensure_ascii=False, indent=4)
    with open(f"{output_dir}/train_valid_data/{language}_val.json", 'w', encoding='utf-8') as output_file:
        json.dump(valid_data, output_file, ensure_ascii=False, indent=4)
    with open(f"{output_dir}/test_data/{language}_test.json", 'w', encoding='utf-8') as output_file:
        json.dump(test_data, output_file, ensure_ascii=False, indent=4)


    # Group the remaining data by multi_label to get test sets
    grouped_by_label = lang_df.groupby('multi_label')
    for label, label_df in grouped_by_label:
        # Skip this as we add all the human samples into the other test sets.
        if label == 'human':
            continue
        

        # Restrict test set size
        test_size = min(TEST_SIZE, len(human_df) * 2, len(label_df) * 2)
        print("Eval set", label, "n =", test_size)
        test_hdf = human_df.sample(n=test_size // 2, random_state=SEED).reset_index(drop=True)
        test_ldf = label_df.sample(n=test_size // 2, random_state=SEED).reset_index(drop=True)
        test_df = pd.concat([test_hdf, test_ldf]).sample(frac=1, random_state=SEED).reset_index(drop=True)

        # Format as list of dicts
        test_data = test_df[["text", "result"]].to_dict(orient="records")  
        with open(f"{output_dir}/test_data/{language}_{label}.json", 'w', encoding='utf-8') as output_file:
            json.dump(test_data, output_file, ensure_ascii=False, indent=4)

        # Extract just the labels and save as torch tensors
        test_labels = torch.tensor(test_df["result"].astype(int).tolist())
        torch.save(test_labels, f"{output_dir}/labels/{language}_{label}.pt")

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
TEST_SIZE = 20

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

# Process test split first
test_split = data[data['split'] == 'test']

# Group by language
grouped_by_language = test_split.groupby('language')

for language, lang_df in grouped_by_language:
    # Get human data for this language
    human_df = lang_df[lang_df['multi_label'] == 'human']

    # Use some of the data for a separate general test split
    if language in ["en", "es", "ru"]:
        lang_df, test_df = train_test_split(
            lang_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=lang_df["result"]
        )
        human_df, test_human_df = train_test_split(
            human_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=human_df["result"]
        )

        # Restrict size to 100 of each
        test_df = test_df.sample(n=(TEST_SIZE // 2), random_state=42)
        test_human_df = test_human_df.sample(n=min(TEST_SIZE // 2, len(test_human_df)), random_state=42)
        test_df = pd.concat([test_df, test_human_df])

        # Format as list of dicts
        test_data = test_df[["text", "result"]].to_dict(orient="records")
        with open(f"{output_dir}/test_data/{language}_test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=4)

        # Extract just the labels and save as torch tensors
        test_labels = torch.tensor(test_df["result"].astype(int).tolist())
        torch.save(test_labels, f"{output_dir}/labels/{language}_test.pt")

    # Group the remaining data by multi_label to get test sets
    grouped_by_label = lang_df.groupby('multi_label')

    for label, label_df in grouped_by_label:
        # Skip this as we add all the human samples into the other test sets.
        if label == 'human':
            continue

        # Restrict size to 100 of each
        test_df = label_df.sample(n=(TEST_SIZE // 2), random_state=42)
        test_human_df = human_df.sample(n=min((TEST_SIZE // 2), len(human_df)), random_state=42)
        # Combine LLM with human text samples
        test_df = pd.concat([test_df, test_human_df])

        # Format as list of dicts
        test_data = test_df[["text", "result"]].to_dict(orient="records")  
        with open(f"{output_dir}/test_data/{language}_{label}.json", 'w', encoding='utf-8') as output_file:
            json.dump(test_data, output_file, ensure_ascii=False, indent=4)

        # Extract just the labels and save as torch tensors
        test_labels = torch.tensor(test_df["result"].astype(int).tolist())
        torch.save(test_labels, f"{output_dir}/labels/{language}_{label}.pt")


# Process train split
train_split = data[data['split'] == 'train']

# Group by language
grouped_by_language = train_split.groupby('language')

for language, language_group in grouped_by_language:
    # Split train into train/val (90/10 of the train set)
    train_df, val_df = train_test_split(
        language_group, 
        test_size=0.1, 
        random_state=42, 
        stratify=language_group["result"]
    )

    # Restrict size to 150 of each
    train_df = train_df.sample(n=TRAIN_SIZE, random_state=42)
    val_df = val_df.sample(n=VALID_SIZE, random_state=42)

    # Output as list of dicts
    train_data = train_df[["text", "result"]].to_dict(orient="records")
    val_data = val_df[["text", "result"]].to_dict(orient="records")
    
    with open(f"{output_dir}/train_valid_data/{language}_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(f"{output_dir}/train_valid_data/{language}_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    # Extract just the labels and save as torch tensors
    train_labels = torch.tensor(train_df["result"].astype(int).tolist())
    val_labels = torch.tensor(val_df["result"].astype(int).tolist())

    torch.save(train_labels, f"{output_dir}/labels/{language}_train.pt")
    torch.save(val_labels, f"{output_dir}/labels/{language}_val.pt")

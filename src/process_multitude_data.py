"""
Parses the MULTITuDE csv into json objects that match the format of the data
expected by the text fluoroscopy pipeline.
"""

import pandas as pd
import json
import os

# Open MULTITuDE csv
output_dir = 'dataset/processed_multitude'
csv_file = 'dataset/raw/multitude.csv'
data = pd.read_csv(csv_file)

# Process test split first
test_split = data[data['split'] == 'test']

# Group by language
grouped_by_language = test_split.groupby('language')

for language, language_group in grouped_by_language:
    # Group by multi_label
    grouped_by_label = language_group.groupby('multi_label')

    for label, label_group in grouped_by_label:
        # Skip this as we add all the human samples into the other test sets.
        if label == 'human':
            continue

        # Output json object of {"text": text, "label": label}
        json_objects = label_group.apply(
            lambda row: {"text": row['text'], "result": row['label']}, axis=1
        ).tolist()

        # Get the human written text samples
        human_text = language_group[language_group['multi_label'] == 'human'].apply(
            lambda row: {"text": row['text'], "result": row['label']}, axis=1
        ).tolist()
        
        # Dump into file f"multitude__{label}.json"
        output_filename = f"multitude_{label}.json"
        output_sub_dir = os.path.join(output_dir, "test_data", str(language))
        os.makedirs(output_sub_dir, exist_ok=True)
        output_path = os.path.join(output_sub_dir, output_filename)  # Replace with your output directory
        
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(json_objects + human_text, output_file, ensure_ascii=False, indent=4)


# Process train split
train_split = data[data['split'] == 'train']

# Group by language
grouped_by_language = train_split.groupby('language')

for language, language_group in grouped_by_language:
    # Output json object of {"text": text, "label": label}
    json_objects = language_group.apply(
        lambda row: {"text": row['text'], "result": row['label']}, axis=1
    ).tolist()
    
    output_filename = f"multitude_{language}_train.json"
    output_sub_dir = os.path.join(output_dir, "train_valid_data")
    os.makedirs(output_sub_dir, exist_ok=True)
    output_path = os.path.join(output_sub_dir, output_filename)  # Replace with your output directory
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(json_objects, output_file, ensure_ascii=False, indent=4)

import pandas as pd 
import json 
from sklearn.model_selection import train_test_split
import os
import torch

filepath = 'dataset/raw/counseling.csv'

df = pd.read_csv(filepath)

# remove empty rows
df = df.dropna(subset=["Response", "LLM"])
df = df[df["Response"].astype(str).str.strip() != ""]
df = df[df["LLM"].astype(str).str.strip() != ""]

# remove these characters because they make it rly easy to distinguish llm (llm uses \n and human uses \u00a0)
df["Response"] = df["Response"].astype(str).str.replace('\u00a0', ' ', regex=False).str.replace('\n', ' ', regex=False)
df["LLM"] = df["LLM"].astype(str).str.replace('\u00a0', ' ', regex=False).str.replace('\n', ' ', regex=False)


# add binary label
human_df = df[["Context", "Response"]].copy()
human_df["result"] = "0"
human_df.rename(columns={"Response": "text"}, inplace=True)

llm_df = df[["Context", "LLM"]].copy()
llm_df["result"] = "1"
llm_df.rename(columns={"LLM": "text"}, inplace=True)

# balancing groups: keep only one row per unique context
human_df = human_df.drop_duplicates(subset=["Context"])
llm_df = llm_df.drop_duplicates(subset=["Context"])

# take 100 from each, ensuring balance and uniqueness
human_df = human_df.sample(n=500, random_state=42)
llm_df = llm_df.sample(n=500, random_state=42)

# combine and shuffle
combined_df = pd.concat([human_df, llm_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# split into test an train (80/20)
# train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df["result"])
train_df, test_df = train_test_split(
    combined_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=combined_df["result"]
)

# Further split train into train/val (90/10 of the train set)
train_df, val_df = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=42, 
    stratify=train_df["result"]
)

# format as list of dicts
train_data = train_df[["text", "result"]].to_dict(orient="records")
val_data = val_df[["text", "result"]].to_dict(orient="records")
test_data = test_df[["text", "result"]].to_dict(orient="records")

# Save to JSON
output_dir = "dataset/processed_counseling"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open(f"{output_dir}/val.json", "w") as f:
    json.dump(val_data, f, indent=4)
with open(f"{output_dir}/test.json", "w") as f:
    json.dump(test_data, f, indent=4)

# Extract just the labels and save as torch tensors
train_labels = torch.tensor(train_df["result"].astype(int).tolist())
val_labels = torch.tensor(val_df["result"].astype(int).tolist())
test_labels = torch.tensor(test_df["result"].astype(int).tolist())

torch.save(train_labels, f"{output_dir}/train.pt")
torch.save(val_labels, f"{output_dir}/val.pt")
torch.save(test_labels, f"{output_dir}/test.pt")


print(f"Saved train.json ({len(train_data)} examples) and test.json ({len(test_data)} examples) and val.json ({len(val_data)}).")

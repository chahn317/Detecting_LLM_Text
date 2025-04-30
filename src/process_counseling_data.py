import pandas as pd 
import json 
from sklearn.model_selection import train_test_split

filepath = 'dataset/raw/counseling.csv'

df = pd.read_csv(filepath)

# remove empty rows
df = df.dropna(subset=["Response", "LLM"])
df = df[df["Response"].astype(str).str.strip() != ""]
df = df[df["LLM"].astype(str).str.strip() != ""]

# add binary label
human_df = df[["Context", "Response"]].copy()
human_df["result"] = "0"
human_df.rename(columns={"Response": "text"}, inplace=True)

llm_df = df[["Context", "LLM"]].copy()
llm_df["result"] = "1"
llm_df.rename(columns={"LLM": "text"}, inplace=True)

# balancing groups
min_len = min(len(human_df), len(llm_df))
human_df = human_df.sample(n=min_len, random_state=42)
llm_df = llm_df.sample(n=min_len, random_state=42)
combined_df = pd.concat([human_df, llm_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# split into test an train (80/20)
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df["result"])

# format as list of dicts
train_data = train_df[["text", "result"]].to_dict(orient="records")
test_data = test_df[["text", "result"]].to_dict(orient="records")

# Save to JSON
with open("dataset/processed_counseling/train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("dataset/processed_counseling/test.json", "w") as f:
    json.dump(test_data, f, indent=4)

print(f"Saved train.json ({len(train_data)} examples) and test.json ({len(test_data)} examples).")

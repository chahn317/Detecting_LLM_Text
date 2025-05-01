from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from sklearn.metrics import auc,roc_curve
import json
import argparse
# it takes approximately 35 seconds when there are 150 pairs

# load data
def load_human_llm_pairs(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    human_text = []
    llm_text = []
    for entry in data:
        label = int(entry["result"])
        text = entry["text"]
        if label == 0:
            human_text.append(text)
        elif label == 1:
            llm_text.append(text)
    print(path)
    print(f"Loaded {len(human_text)} human texts and {len(llm_text)} LLM texts.")
    return human_text, llm_text


def get_auc(human_preds, llm_preds):
    fpr, tpr, _ = roc_curve([0] * len(human_preds) + [1] * len(llm_preds), human_preds + llm_preds,pos_label=1)
    roc_auc = auc(fpr, tpr)
    return float(round(roc_auc, 4))


def run(args):
    device = "cpu"
    detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
    detector = AutoModelForSequenceClassification.from_pretrained(detector_path_or_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(detector_path_or_id, cache_dir=args.cache_dir)
    detector.eval()
    detector.to(device)
    
    human_text, llm_text = load_human_llm_pairs(args.dataset_path)
    with torch.no_grad():
        inputs = tokenizer(human_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        human_preds = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()

    with torch.no_grad():
        inputs = tokenizer(llm_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        llm_preds = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()

    auroc = get_auc(human_preds, llm_preds)
    print("Detection AUROC: ", auroc)

    # Append result as a json object to the results file
    results = {
        "dataset_path": args.dataset_path,
        "auroc": auroc
    }

    try:
        with open(args.results_path, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = []

    existing_results.append(results)

    with open(args.results_path, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="dataset/processed_data/test_data/pub_claude3.json")
    parser.add_argument('--results-path', type=str, default="results/radar.json")
    parser.add_argument('--cache-dir', type=str, default="/scratch/text-fluoroscopy/.cache")
    args = parser.parse_args()
    run(args)
import transformers
import torch
import torch.nn.functional as F
from sklearn.metrics import auc,roc_curve
import json
# it takes approximately 35 seconds when there are 150 pairs


device = "cpu"
detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
detector = transformers.AutoModelForSequenceClassification.from_pretrained(detector_path_or_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(detector_path_or_id)
detector.eval()
detector.to(device)

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


for dataset_name in ["pub"] :
    for model_name in ["claude3"]:
        path = "dataset/" + dataset_name  + "/" + dataset_name + "_" + model_name + ".json"
        output_path = "output/" + dataset_name  + "/" + dataset_name + "_" + model_name + ".json"
        human_text, llm_text = load_human_llm_pairs(path)

        print("start calculating probabilities for human-written texts")
        with torch.no_grad():
            inputs = tokenizer(human_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            human_preds = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()

        print("start calculating probabilities for LLM-generated texts")
        with torch.no_grad():
            inputs = tokenizer(llm_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k:v.to(device) for k,v in inputs.items()}
            llm_preds = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()


        print("Detection AUROC: ", get_auc(human_preds,llm_preds))

        # store outputs
        output = {
            "human_probs": human_preds,
            "llm_probs": llm_preds
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


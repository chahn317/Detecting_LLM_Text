import torch
import json
import argparse
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic
from sklearn.metrics import auc,roc_curve
from scipy.stats import norm

# Considering balanced classification that p(D0) equals to p(D1), we have
#   p(D1|x) = p(x|D1) / (p(x|D1) + p(x|D0))
def compute_prob_norm(x, mu0, sigma0, mu1, sigma1):
    pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
    pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
    prob = pdf_value1 / (pdf_value0 + pdf_value1)
    return prob

class FastDetectGPT:
    def __init__(self, args):
        self.args = args
        self.criterion_fn = get_sampling_discrepancy_analytic
        self.scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
        self.scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
        self.scoring_model.eval()
        if args.sampling_model_name != args.scoring_model_name:
            self.sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
            self.sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
            self.sampling_model.eval()
        # To obtain probability values that are easy for users to understand, we assume normal distributions
        # of the criteria and statistic the parameters on a group of dev samples. The normal distributions are defined
        # by mu0 and sigma0 for human texts and by mu1 and sigma1 for AI texts. We set sigma1 = 2 * sigma0 to
        # make sure of a wider coverage of potential AI texts.
        # Note: the probability could be high on both left side and right side of Normal(mu0, sigma0).
        #   gpt-j-6B_gpt-neo-2.7B: mu0: 0.2713, sigma0: 0.9366, mu1: 2.2334, sigma1: 1.8731, acc:0.8122
        #   gpt-neo-2.7B_gpt-neo-2.7B: mu0: -0.2489, sigma0: 0.9968, mu1: 1.8983, sigma1: 1.9935, acc:0.8222
        #   falcon-7b_falcon-7b-instruct: mu0: -0.0707, sigma0: 0.9520, mu1: 2.9306, sigma1: 1.9039, acc:0.8938
        distrib_params = {
            'gpt-j-6B_gpt-neo-2.7B': {'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731},
            'gpt-neo-2.7B_gpt-neo-2.7B': {'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935},
            'falcon-7b_falcon-7b-instruct': {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039},
        }
        key = f'{args.sampling_model_name}_{args.scoring_model_name}'
        self.classifier = distrib_params[key]
    # compute conditional probability curvature
    def compute_crit(self, text):
        tokenized = self.scoring_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]
            if self.args.sampling_model_name == self.args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.sampling_tokenizer(text, truncation=True, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.sampling_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
        return crit, labels.size(1)
    # compute probability
    def compute_prob(self, text):
        crit, ntoken = self.compute_crit(text)
        mu0 = self.classifier['mu0']
        sigma0 = self.classifier['sigma0']
        mu1 = self.classifier['mu1']
        sigma1 = self.classifier['sigma1']
        prob = compute_prob_norm(crit, mu0, sigma0, mu1, sigma1)
        return prob, crit, ntoken

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
    detector = FastDetectGPT(args)
    human_text, llm_text = load_human_llm_pairs(args.dataset_path)
    human_preds = []
    with torch.no_grad():
        for ht in human_text:
            prob, crit, ntokens = detector.compute_prob(ht)
            human_preds.append(prob)
    llm_preds = []
    with torch.no_grad():
        for lt in llm_text:
            prob, crit, ntokens = detector.compute_prob(lt)
            llm_preds.append(prob)

    auroc = get_auc(human_preds,llm_preds)
    print("Detection AUROC: ", auroc)

    # Append result as a json object to the results file
    results = {
        "sampling_model_name": args.sampling_model_name,
        "scoring_model_name": args.scoring_model_name,
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
    parser.add_argument('--sampling-model-name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring-model-name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset-path', type=str, default="dataset/processed_data/test_data/pub_claude3.json")
    parser.add_argument('--results-path', type=str, default="results/baselines.json")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="/scratch/text-fluoroscopy/.cache")
    args = parser.parse_args()
    run(args)
# Detecting LLM-Generated Text via Intrinsic Features

This repository contains our reproduction and multilingual extension of the paper:  
**Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features (EMNLP 2024)**

We reproduce the original method and extend it to evaluate performance across 10+ languages. Our implementation includes embedding extraction, KL-based layer selection, and classifier training and evaluation.

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- Transformers (Hugging Face)
- scikit-learn
- tqdm

To install dependencies:

```bash
pip install -r requirements.txt
```

## Hardware Requirements

- A GPU with **at least 11GB of memory** is recommended.
- Input sequences are limited to **300 tokens** to avoid out-of-memory issues.
- For GPUs with less memory, reduce the `max_length` parameter in the relevant scripts.

## Repository Structure

```
.
├── dataset/                     
│   ├── processed_data/           # Data and labels for original English experiment
│   ├── labels/                   # Label files for original datasets
│   ├── processed_counseling/     # Data for counseling dataset experiments
│   └── processed_multitude/      # Data for multilingual extension
├── src/
│   ├── process_multitude_data.py # Preprocessing script for multilingual input
|   └── process_counseling_data.py # Preprocessing script for counseling input
├── base.py                        # Main python file for English-only experiments
├── multilingual.py                # python file for multilingual experiments
├── main_other_data.ipynb         # Notebook for multilingual and additional datasets
├── process_text_length.ipynb     # Helper notebook for analyzing token lengths
└── figures/                      # Stores plots such as AUROC vs. layer
```

## Running the Experiments

### Baselines

To run the baseline methods RADAR and FastDetectGPT, first set up the environments for each of the methods. The requirements files can be found in each of their respective directories.
```
─ fast-detect-gpt
   └── requirements.txt            # Requirements file
─ radar
   └── env
      ├── radar_core.yaml         # YAML file for conda environment
      └── radar_requirements.txt  # Requirements file
```

Then, run the following script to save the evaluation results to the `results` directory.
```bash
./scripts/baselines.sh
```

Alternatively, run the Python scripts for RADAR and FastDetectGPT respectively. An example usage is as such
```bash
python radar/RADAR.py \
  --dataset-path "dataset/processed_data/test_data/pub_gpt4.json" \
  --results-path "results/radar_multitude.json" \
  --cache-dir "/scratch/text-fluoroscopy/.cache"

python fast-detect-gpt/src/replication.py \
  --dataset-path "dataset/processed_data/test_data/pub_gpt4.json" \
  --results-path "results/fast-detect-gpt.json" \
  --cache-dir "/scratch/text-fluoroscopy/.cache"
```

#### Explanation of Arguments

- `--dataset-path`: Specifies the path to the data to evaluate.
- `--results-path`: Specifies the path to the file to save the results in.
- `--cache-dir`: Specifies the path to the cache for the models.

### Base (English-only)

To reproduce the English-only experiment across all layers, run the following command

```bash
python base.py --layer_num -1 --datasets pub writing xsum
```

#### Explanation of Arguments

- `--layer_num`: Specifies which layer's embeddings to use for training and evaluation.
  - `0–32`: Use embeddings from that specific layer.
  - `-1`: Use the layer with the maximum KL divergence (automatically selected per example).
  - `-2`: Run the experiment across all layers (0–32) and evaluate each one.

- `--datasets`: Specifies the test datasets to evaluate the classifier on. Each dataset corresponds to a different domain or type of text:
  - `pub`: Public domain or general web content
  - `writing`: Formal or academic writing samples
  - `xsum`: News articles and summaries from the XSum dataset

### Text Length + Cross-Domain Extension

For our text length/cross-domain extensions, run the code in `text_length.ipynb` and `cross_domain.ipynb`, respectively.

### Multilingual Extension

To run multilingual evaluation with KL-based layer selection:

1. Preprocess data:

```bash
python src/process_multitude_data.py
```

2. Run the following command: 

```bash
python python multilingual.py --layer_num -1 --epochs 10 --dropout 0.4 --lr 0.0003 --language 'cs'
```
#### Explanation of Arguments

- `--layer_num`: Specifies which layer's embeddings to use for training and evaluation.
  - `0–32`: Use embeddings from that specific layer.
  - `-1`: Use the layer with the maximum KL divergence (automatically selected per example).
  - `-2`: Run the experiment across all layers (0–32) and evaluate each one.

- `--epochs`: Number of epochs for classifier training.
- `--dropout`: Dropout probability for the classifier model.
- `--lr`: Learning rate for the optimizer.
- `--language`: Target language code for the experiment. Supported language codes are:
  - English (en)
  - Arabic (ar)
  - Catalan (ca)
  - Czech (cs)
  - German (de)
  - Dutch (nl)
  - Portuguese (pt)
  - Russian (ru)
  - Ukrainian (uk)
  - Chinese (zh)
  - Spanish (es)


## Results & Visualization

- AUROC vs. layer number plots are stored in the `figures/` directory.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{yu2024text,
  title={Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features},
  author={Yu, Xiao and Chen, Kejiang and Yang, Qi and Zhang, Weiming and Yu, Nenghai},
  booktitle={EMNLP},
  year={2024}
}
```

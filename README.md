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
├── main.ipynb                    # Main notebook for English-only experiments
├── main_other_data.ipynb         # Notebook for multilingual and additional datasets
├── process_text_length.ipynb     # Helper notebook for analyzing token lengths
└── figures/                      # Stores plots such as AUROC vs. layer
```

## Running the Experiments

### Base (English-only)

To reproduce the English-only experiment across all layers, run the steps in:

- `main.ipynb`

### Multilingual Extension

To run multilingual evaluation with KL-based layer selection:

1. Preprocess data:

```bash
python src/process_multitude_data.py
```

2. Run classification and evaluation by modifying parameters (e.g., language, file paths) in:

- `main_other_data.ipynb`

> Note: We are currently working on converting these notebooks into fully scripted pipelines to simplify reproducibility.

## Supported Languages

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

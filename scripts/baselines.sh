#!/bin/bash
# Usage: ./scripts/baselines.sh

datasets=("pub" "writing" "xsum")
models=("claude3" "gpt3.5" "gpt4")
results_path="results/fast-detect-gpt.json"

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        path="dataset/processed_data/test_data/${dataset}_${model}.json"
        python fast-detect-gpt/src/replication.py \
            --dataset-path "$path" \
            --results-path "$results_path"
    done
done


# Run baseline on MULTITuDE
languages=("en" "es")
models=("alpaca-lora-30b" "gpt-3.5-turbo" "gpt-4")
results_path="results/fast-detect-gpt_multitude.json"

for lang in "${languages[@]}"; do
    for model in "${models[@]}"; do
        path="dataset/processed_multitude/test_data/${lang}/multitude_${model}.json"
        python fast-detect-gpt/src/replication.py \
            --dataset-path "$path" \
            --results-path "$results_path"
    done
done
#!/bin/bash
# Usage: ./scripts/baselines.sh

cache_dir="/scratch/text-fluoroscopy/.cache"

datasets=("pub" "writing" "xsum")       # Modify this list as necessary
models=("claude3" "gpt3.5" "gpt4")      # Modify this list as necessary

# Run baselines on the LLM detection data data from the text fluoroscopy paper
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        path="dataset/processed_data/test_data/${dataset}_${model}.json"
        python fast-detect-gpt/src/replication.py \
            --dataset-path "$path" \
            --results-path "results/fast-detect-gpt.json" \
            --cache-dir "$cache_dir"
        python radar/RADAR.py \
            --dataset-path "$path" \
            --results-path "results/radar.json" \
            --cache-dir "$cache_dir"
    done
done

# Run baselines on MULTITuDE
languages=("en" "es")                   # Modify this list as necessary
models=("alpaca-lora-30b" "gpt-3.5-turbo" "gpt-4")  # Modify this list as necessary

for lang in "${languages[@]}"; do
    for model in "${models[@]}"; do
        path="dataset/processed_multitude/test_data/${lang}/multitude_${model}.json"
        python fast-detect-gpt/src/replication.py \
            --dataset-path "$path" \
            --results-path "results/fast-detect-gpt_multitude.json" \
            --cache-dir "$cache_dir"
        python radar/RADAR.py \
            --dataset-path "$path" \
            --results-path "results/radar_multitude.json" \
            --cache-dir "$cache_dir"
    done
done
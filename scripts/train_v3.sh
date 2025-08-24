#!/usr/bin/env bash
set -euo pipefail


python -m adtecplayground.train \
--model_name "tohoku-nlp/bert-base-japanese-v3" \
--dataset_name "cyberagent/AdTEC" \
--dataset_config "ad-acceptability" \
--max_length 128 \
--seeds 0 1 2 3 4 5 6 7 8 9 \
--learning_rates 2e-5 5.5e-5 2e-6 \
--num_train_epochs 30 \
--patience 3 \
--batch_size 32 \
--warmup_ratio 0.0 \
--weight_decay 0.0 \
--gradient_accumulation_steps 1 \
--output_dir ./runs/adtec-bert-v3 \
--fp16
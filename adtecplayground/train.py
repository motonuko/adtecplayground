import os
import json
import random
import argparse

import numpy as np
import torch
from torch.optim import Adam
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, set_seed

from adtecplayground.config import AdtecConfig
from adtecplayground.data import load_splits, build_tokenizer, encode_dataset
from adtecplayground.metrics import compute_metrics
from adtecplayground.modeling import build_model


def set_all_seeds(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AdamTrainer(Trainer):
    def __init__(self, *args, lr: float, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = lr


def create_optimizer(self):
    if self.optimizer is None:
        self.optimizer = Adam(self.model.parameters(), lr=self._lr)
    return self.optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=AdtecConfig.model_name)
    parser.add_argument("--dataset_name", type=str, default=AdtecConfig.dataset_name)
    parser.add_argument("--dataset_config", type=str, default=AdtecConfig.dataset_config)
    parser.add_argument("--max_length", type=int, default=AdtecConfig.max_length)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--learning_rates", type=float, nargs="*", default=[2e-5, 5.5e-5, 2e-6])
    parser.add_argument("--num_train_epochs", type=int, default=AdtecConfig.num_train_epochs)
    parser.add_argument("--patience", type=int, default=AdtecConfig.patience)
    parser.add_argument("--batch_size", type=int, default=AdtecConfig.batch_size)
    parser.add_argument("--warmup_ratio", type=float, default=AdtecConfig.warmup_ratio)
    parser.add_argument("--weight_decay", type=float, default=AdtecConfig.weight_decay)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=AdtecConfig.gradient_accumulation_steps)
    parser.add_argument("--output_dir", type=str, default=AdtecConfig.output_dir)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=AdtecConfig.save_total_limit)
    args = parser.parse_args()
    raw, label2id, id2label = load_splits(args.dataset_name, args.dataset_config)
    tok = build_tokenizer(args.model_name)
    enc, collator = encode_dataset(raw, tok, args.max_length)

    best_run = None
    results = []
    for seed in args.seeds:
        set_all_seeds(seed)
        for lr in args.learning_rates:
            run_id = f"seed{seed}-lr{lr}"
            out_dir = os.path.join(args.output_dir, run_id)
            os.makedirs(out_dir, exist_ok=True)

            model = build_model(args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id)

            targs = TrainingArguments(
                output_dir=out_dir,
                learning_rate=lr,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.num_train_epochs,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                seed=seed,
                warmup_ratio=args.warmup_ratio,
                weight_decay=args.weight_decay,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                fp16=args.fp16,
                save_total_limit=args.save_total_limit,
                dataloader_pin_memory=True,
                dataloader_num_workers=int(os.environ.get("NUM_WORKERS", 2)),
                log_level="info",
                report_to=["none"],
            )

            trainer = AdamTrainer(
                model=model,
                args=targs,
                train_dataset=enc["train"],
                eval_dataset=enc["valid"],
                tokenizer=tok,
                data_collator=collator,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
                lr=lr,
            )

            trainer.train()
            eval_metrics = trainer.evaluate(enc["valid"])  # best by f1
            test_metrics = trainer.evaluate(enc["test"])  # report

            summary = {
                "run": run_id,
                "best_ckpt": trainer.state.best_model_checkpoint,
                "val_accuracy": eval_metrics.get("eval_accuracy"),
                "val_f1": eval_metrics.get("eval_f1"),
                "test_accuracy": test_metrics.get("eval_accuracy"),
                "test_f1": test_metrics.get("eval_f1"),
            }
            results.append(summary)
            if best_run is None or summary["val_f1"] > best_run["val_f1"]:
                best_run = summary

    print("===== Runs Summary (sorted by val_f1) =====")
    for row in sorted(results, key=lambda x: x["val_f1"], reverse=True):
        print(row)

    print("===== Selected Best Run (by val_f1) =====")
    print(best_run)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"runs": results, "best": best_run, "label2id": label2id, "id2label": id2label}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

import json
import re
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from math import isnan

# ----- Helpers -----
seed_lr_pattern = re.compile(r"seed(?P<seed>\d+)-lr(?P<lr>[\d\.eE\-]+)")

def infer_model_name_from_path_or_ckpt(path: Path, summary_obj: dict) -> str:
    path_str = str(path).replace("\\", "/")
    m = re.search(r"runs/([^/]+)/summary\.json", path_str)
    if m:
        return m.group(1)
    for r in summary_obj.get("../runs", []):
        ckpt = (r.get("best_ckpt", "") or "").replace("\\", "/")
        mm = re.search(r"runs/([^/]+)/", ckpt)
        if mm:
            return mm.group(1)
    return path.stem

def parse_lr(lr_str: str) -> float:
    try:
        return float(lr_str)
    except Exception:
        return float("nan")

def load_records(candidate_paths, metric_key):
    records = []
    for p in candidate_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        model_name = infer_model_name_from_path_or_ckpt(p, summary)
        for r in summary.get("runs", []):
            run_id = r.get("run", "")
            m = seed_lr_pattern.fullmatch(run_id)
            if not m:
                continue
            seed = int(m.group("seed"))
            lr_str = m.group("lr")
            lr = parse_lr(lr_str)
            if metric_key not in r:
                continue
            records.append(
                {"model": model_name, "lr_str": lr_str, "lr": lr,
                 "seed": seed, "metric": float(r[metric_key])}
            )
    return records

def plot_boxplot(records, metric_key, out_path: Path):
    groups = defaultdict(list)
    for rec in records:
        key = (rec["model"], rec["lr_str"], rec["lr"])
        groups[key].append(rec["metric"])
    def sort_key(t):
        model, lr_str, lr_val = t
        return (model, float("inf") if isnan(lr_val) else lr_val)
    sorted_keys = sorted(groups.keys(), key=sort_key)
    data = [groups[k] for k in sorted_keys]
    labels = [f"{k[0]} | lr={k[1]}" for k in sorted_keys]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))
    ax.boxplot(data, vert=True, patch_artist=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Distribution of {metric_key} by Model × Learning Rate (per seeds)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved figure to {out_path}")

def main():
    metric_key = sys.argv[1] if len(sys.argv) > 1 else "test_f1"
    candidate_paths = [
        Path("runs/adtec-bert-v2/summary.json"),
        Path("runs/adtec-bert-v3/summary.json"),
    ]
    records = load_records(candidate_paths, metric_key)
    if not records:
        print("No usable records found. Check file paths and metric name.")
        sys.exit(1)
    out_path = Path(f"runs/boxplot_{metric_key}.png")
    plot_boxplot(records, metric_key, out_path)

if __name__ == "__main__":
    main()

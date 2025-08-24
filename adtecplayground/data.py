from typing import Dict, List, Tuple
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding


def load_splits(dataset_name: str, dataset_config: str) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str]]:

    raw = load_dataset(dataset_name, dataset_config)
    assert all(k in raw for k in ["train", "valid", "test"])  # expected splits

    label_list = sorted(set(raw["train"]["label"]) | set(raw["valid"]["label"]) | set(raw["test"]["label"]))
    label2id = {name: i for i, name in enumerate(label_list)}
    id2label = {i: name for name, i in label2id.items()}
    return raw, label2id, id2label


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _preprocess_fn(tokenizer, max_length: int):
    def fn(examples):
        return tokenizer(examples["title"], truncation=True, max_length=max_length)
    return fn


def encode_dataset(raw: DatasetDict, tokenizer, max_length: int):
    encoded = {}
    for split in ["train", "valid", "test"]:
        enc = raw[split].map(_preprocess_fn(tokenizer, max_length), batched=True,
                             remove_columns=[c for c in raw[split].column_names if c not in ("label",)])
        enc = enc.map(lambda x: {"labels": 1 if x["label"] == "unacceptable" else 0}, remove_columns=["label"])  # stable map
        encoded[split] = enc
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return encoded, collator

from dataclasses import dataclass, field
from typing import List


@dataclass
class AdtecConfig:
    model_name: str = "tohoku-nlp/bert-base-japanese-v2"
    dataset_name: str = "cyberagent/AdTEC"
    dataset_config: str = "ad-acceptability"

    max_length: int = 128
    seeds: List[int] = field(default_factory=lambda: [0])
    learning_rates: List[float] = field(default_factory=lambda: [2e-5, 5.5e-5, 2e-6])

    num_train_epochs: int = 30
    patience: int = 3
    batch_size: int = 32
    warmup_ratio: float = 0.0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1
    output_dir: str = "./runs/adtec-bert-v2"
    fp16: bool = True
    save_total_limit: int = 2

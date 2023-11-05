from datetime import datetime
from typing import Literal, TypedDict

class Config(TypedDict, total=False):
    lr: float = 1e-2
    num_epochs: int = 500
    seed: int = 42
    batch_size: int = 3
    target_level: int = 3
    num_workers: int = 10
    base_dir: str = "/data2/projects/DigiStrudMed_sklein/"
    overfit: float = 1.0
    run_name: str = __file__.split("/")[-1].split(".")[0]
    data_split: list[float] = [0.8, 0.15, 0.05]
    annos_of_interest: list[str] = ["Tissue", "Tumor_vital", "Angioinvasion", "Tumor_necrosis", "Tumor_regression"]
    gradient_accumulation_steps: int = 48
    date: Literal[datetime.now().strftime("%Y-%m-%d")] = datetime.now().strftime("%Y-%m-%d")
    output_dir: str
    job_name: str
    group_name: str
    swa: int
    overfit: float

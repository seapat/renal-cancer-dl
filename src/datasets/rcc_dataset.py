from src.nanorcc.parse import get_rcc_data
from src.nanorcc.preprocess import (
    CodeClassGeneSelector,
    FunctionGeneSelector,
    Normalize,
)
from src.nanorcc.quality_control import QualityControl

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def rcc_to_csv(dir_path, save_path=None, housekeeping: bool = True):
    """Convert RCC file to csv file.

    Args:
        dir_path (str): Path to RCC file.

    Returns:
        str: Path to csv file.
    """

    if not dir_path.endswith("/"):
        dir_path = dir_path + "/"

    # read the data files
    counts, genes = get_rcc_data(f"{dir_path}*RCC")

    # perform Quality Control check and remove samples that fail Quality Control
    qc = QualityControl()
    counts = qc.drop_failed_qc(counts)

    # pass genes to CodeClassGeneSelector for easy gene selection during normalization.
    ccgs = CodeClassGeneSelector(genes)

    # initialize a Normalize object on the raw data
    norm = Normalize(counts, genes)

    # steps: subtract background, adjust counts by positive controls, adjust counts by
    # housekeeping genes
    # a pipeline for normalization
    normalized_df = (
        norm.background_subtract(genes=ccgs.get("Negative"), drop_genes=True)
        .scale_by_genes(genes=ccgs.get("Positive"), drop_genes=True)
        .scale_by_genes(genes=ccgs.get("Housekeeping"), drop_genes=housekeeping)
    ).norm_data

    if save_path is not None:
        if not save_path.endswith("/"):
            save_path = save_path + "/"
        normalized_df.to_csv(f"{save_path}normalized_counts.csv")

    return normalized_df


class RCCDataset(Dataset):
    def __init__(
        self, rcc_path: str, surv_path: str, sparse: int | None = None
    ) -> None:
        normalized_df: pd.DataFrame = rcc_to_csv(rcc_path)

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns, index=normalized_df.index)

        tabular: pd.DataFrame = pd.read_csv(surv_path)
        tabular.index = tabular["case"]  # type: ignore

        self.data = pd.merge(df_scaled, tabular, left_index=True, right_index=True)
        self.nonscaled_data = pd.merge(normalized_df, tabular, left_index=True, right_index=True)


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, index
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, list[str]]:
        # get the features and target for a specific index
        sample = torch.Tensor(self.data.iloc[index])
        features = sample[:-4]
        case, surv_days, death, uncensored = sample[-4:]
        genes = list(self.data.columns[:-4])
        sample_id = self.data.iloc[index].name
        assert case == sample_id or case == 0, "mismatch of survival data and rcc data"

        return features, surv_days, uncensored, sample_id, genes


if __name__ == "__main__":
    dir_path = (
        "/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/NanoString_RCC/"
    )
    save_path = "/data2/projects/DigiStrudMed_sklein/"
    rcc_to_csv(dir_path, save_path)

    dataset = RCCDataset(
        "/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/NanoString_RCC/",
        "/data2/projects/DigiStrudMed_sklein/" + "survival_status.csv",
    )

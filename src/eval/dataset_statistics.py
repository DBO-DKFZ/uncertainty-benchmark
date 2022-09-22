import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..utils.system import RC_DICT, insert_rc_args


def show_slide_statistics(slide_df: pd.DataFrame, fname: str = None):
    """
    Method to plot slide statistics
    slide_df has columns    slide_name  tumor   non_tumor   total   frac
    """
    sns.set_context(context=None, font_scale=1, rc=RC_DICT)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(15, 5))
    index = np.arange(1, len(slide_df) + 1)
    non_tumor = slide_df["non_tumor"].to_numpy()
    tumor = slide_df["tumor"].to_numpy()
    ax.bar(index, non_tumor, label="Non-Tumor")  # , color="magenta")
    ax.bar(index, tumor, bottom=non_tumor, label="Tumor")  # , color="cyan")
    ax.legend()
    ax.set_xlim([0, len(slide_df) + 1])
    ax.set_xlabel("Slide index")
    ax.set_ylabel("Number of tiles")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def show_center_statistics(id_df: pd.DataFrame, ood2_df: pd.DataFrame, ood4_df: pd.DataFrame, fname: str = None):
    """
    Method to plot center statistics
    dataframes have columns    slide_name	patch_path	label	tumor_coverage	x_pos	y_pos	patch_size
    """
    sns.set_context(context=None, font_scale=1, rc=RC_DICT)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(7, 5))

    id_tumor = len(id_df[id_df["label"] == "tumor"])
    ood2_tumor = len(ood2_df[ood2_df["label"] == "tumor"])
    ood4_tumor = len(ood4_df[ood4_df["label"] == "tumor"])

    data = [
        ["ID centers", "Non-Tumor", len(id_df) - id_tumor],
        ["ID centers", "Tumor", id_tumor],
        ["OOD (center 2)", "Non-Tumor", len(ood2_df) - ood2_tumor],
        ["OOD (center 2)", "Tumor", ood2_tumor],
        ["OOD (center 4)", "Non-Tumor", len(ood4_df) - ood4_tumor],
        ["OOD (center 4)", "Tumor", ood4_tumor],
    ]
    df = pd.DataFrame(data, columns=["center", "label", "count"])

    ax = sns.barplot(x="center", y="count", hue="label", data=df, saturation=1)
    ax.legend(title=None)
    ax.set_xlabel(None)
    ax.set_ylabel("Number of tiles")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def show_tile_statistics(
    id_df: pd.DataFrame, ood2_df: pd.DataFrame, ood4_df: pd.DataFrame, rc_args: dict = None, fname: str = None
):
    """
    Method to plot tile statistics
    dataframes have columns    slide_name	patch_path	label	tumor_coverage	x_pos	y_pos	patch_size
    """
    sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    id_tumor = id_df[id_df["label"] == "tumor"]
    ood2_tumor = ood2_df[ood2_df["label"] == "tumor"]
    ood4_tumor = ood4_df[ood4_df["label"] == "tumor"]

    palette = sns.color_palette("colorblind", 2)
    axs[0].hist(id_tumor["tumor_coverage"].to_numpy(), color=palette[-1])
    axs[0].set_title("ID centers")
    axs[1].hist(ood2_tumor["tumor_coverage"].to_numpy(), color=palette[-1])
    axs[1].set_title("OOD (center 2)")
    axs[2].hist(ood4_tumor["tumor_coverage"].to_numpy(), color=palette[-1])
    axs[2].set_title("OOD (center 4)")
    for ax in axs:
        ax.set_xlabel("Tumor coverage")
    axs[0].set_ylabel("Number of tiles")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def create_df(slide_folders: list, slide_indices: list):
    dfs = []
    for idx in slide_indices:
        fol = slide_folders[idx]
        csv_p = fol / ("tile_information.csv")
        assert csv_p.exists()
        dfs.append(pd.read_csv(csv_p))
    tile_inf = pd.concat(dfs)
    tile_inf = tile_inf.reset_index(drop=True)
    return tile_inf


if __name__ == "__main__":
    data_path = Path(os.environ["DATASET_LOCATION"]) / "Camelyon17" / "tiles"

    slide_inf_path = data_path / "slide_information.csv"
    slide_inf = pd.read_csv(slide_inf_path)

    slide_folders = [fol.absolute() for fol in data_path.iterdir() if fol.is_dir()]
    slide_folders = sorted(slide_folders, key=lambda x: x.parts[-1])
    id_indices = [*range(0, 10), *range(10, 20), *range(30, 40)]
    ood2_indices = [*range(20, 30)]
    ood4_indices = [*range(40, 50)]

    id_df = create_df(slide_folders, id_indices)
    ood2_df = create_df(slide_folders, ood2_indices)
    ood4_df = create_df(slide_folders, ood4_indices)

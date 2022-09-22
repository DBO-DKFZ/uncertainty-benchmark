import os
import sys
from typing import Union
from pathlib import Path

from tqdm import tqdm
import torch
import torchmetrics
from torchmetrics import (
    Accuracy,
    F1Score,
    ConfusionMatrix,
    PrecisionRecallCurve,
    Precision,
    Recall,
    AUROC,
    AUC,
    AveragePrecision,
    CalibrationError,
)
from torchmetrics.functional import accuracy
import numpy as np
import matplotlib

# print(matplotlib.get_backend())
# matplotlib.use("Agg")  # Fixed issue when script was stuck at import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import seaborn as sns

from ..datamodules.camelyon_datamodules import *
from ..metrics.metrics import compute_confidence, normed_entropy
from ..utils.system import RC_DICT, insert_rc_args
from .visualizations_base import (
    Run,
    EnsembleRun,
    create_runs_from_folder,
    create_ensemble_from_folder,
    extract_results,
)


def compute_metrics(
    runs,
    split,
    metrics=None,
    compute_significance=False,
    plot_x="split",
    plot_y="value",
    plot_hue="method",
    plot_col="metric",
    rc_args: dict = None,
    color_palette: str = "colorblind",
    fname: str = None,
):

    if isinstance(split, str):
        splits = [split]
    else:
        splits = split

    results = []

    for split in splits:

        # print(f"Computing {split} metrics.")

        for run in runs:

            if split == "val":
                run_preds = run.val_preds
            elif split == "test_id":
                run_preds = run.test_id_preds
            elif split == "test_ood":
                run_preds = run.test_ood_preds
            elif split == "test_ood2":
                run_preds = run.test_ood2_preds
            elif split == "test_ood4":
                run_preds = run.test_ood4_preds
            else:
                raise RuntimeError(f"Undefined split {split}")
            out_sm, label = extract_results(run_preds)

            # Remap split for nice reading
            print_split = {
                "val": "validation",
                "test_id": "ID centers",
                "test_ood": "OOD centers",
                "test_ood2": "Center 2",
                "test_ood4": "Center 4",
            }[split]

            metric_dict = {}
            metric_dict["method"] = run.name
            metric_dict["split"] = print_split
            metric_dict["run"] = run.exp_path

            if hasattr(run, "properties"):
                for key, value in run.properties.items():
                    metric_dict[key] = value

            if "Accuracy" in metrics:
                acc = Accuracy()(out_sm, label)
                metric_dict["Accuracy"] = acc.item()

            if "Balanced Accuracy" in metrics:
                b_acc = Accuracy(average="macro", num_classes=2)(out_sm, label)
                metric_dict["Balanced Accuracy"] = b_acc.item()

            if "F1" in metrics:
                f1 = F1Score(num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["F1"] = f1.item()

            if "AUROC" in metrics:
                auroc = AUROC(pos_label=1, average="micro")(out_sm[:, 1], label)
                metric_dict["AUROC"] = auroc.item()

            if "AUCPR" in metrics:
                aucpr = AveragePrecision(pos_label=1, average="micro")(out_sm[:, 1], label)
                metric_dict["AUCPR"] = aucpr.item()

            if "Precision" in metrics:
                pre = Precision(num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["Precision"] = pre.item()

            if "Recall" in metrics:
                rec = Recall(num_classes=2, ignore_index=0)(out_sm, label)
                metric_dict["Recall"] = rec.item()

            if "ECE" in metrics:
                ece = CalibrationError(n_bins=20, norm="l1", compute_on_step=True)
                ece.update(out_sm[:, 1], label)
                ece = ece.compute()
                metric_dict["ECE"] = ece.item()

            # if "ACE" in metrics:
            #     ace = AdaptiveCalibrationError()
            #     ace.update(out_sm[:, 1], label)
            #     ace = ace.compute()
            #     metric_dict["ACE"] = ace.item()

            results.append(metric_dict)

    res_df = pd.DataFrame(results)

    to_keep = [col for col in list(res_df.columns) if col not in metrics]

    melted_df = res_df.melt(id_vars=to_keep, var_name="metric", value_name="value")

    sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)
    grid = sns.catplot(
        kind="box",
        x=plot_x,
        y=plot_y,
        hue=plot_hue,
        col=plot_col,
        col_wrap=min(3, len(metrics)),
        data=melted_df,
        # estimator=np.median,
        showfliers=False,
        sharey=False,
        legend_out=True,
    )
    grid.figure.set_figwidth(15)
    grid.figure.set_figheight(5)
    grid.legend.set_title("Method")
    ax1 = grid.axes[0]
    ax1.set_title("Accuracy")
    ax1.set_xlabel(None)
    ax1.set_ylabel("Value")
    ax2 = grid.axes[1]
    ax2.set_title("Balanced Accuracy")
    ax2.set_xlabel(None)
    ax3 = grid.axes[2]
    ax3.set_title("ECE")
    ax3.set_xlabel(None)

    grouped_results = res_df.groupby(by=["split", "method"])

    mean_var = grouped_results.agg(lambda x: (np.mean(x), np.std(x)))

    def latex_meanvar_formatter(x):
        mean, var = x
        return f"$ {mean:.4f}_{{\\pm{var:.4f}}}$"

    # print(mean_var.to_latex(formatters=[latex_meanvar_formatter] * len(mean_var.columns), escape=False))

    if compute_significance:
        raise RuntimeError("Not correctly implemented")

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

    return res_df


def compute_metrics_label_noise(
    runs,
    split,
    metrics=(
        "Accuracy",
        "Balanced Accuracy",
        "AUROC",
        "AUCPR",
        "Precision",
        "Recall",
        "ECE",
        "F1",
    ),
    compute_significance=False,
    rc_args: dict = None,
    color_palette: str = "colorblind",
    fname: str = None,
):

    if isinstance(split, str):
        splits = [split]
    else:
        splits = split

    results = {}

    for split in splits:

        print(f"Computing {split} metrics.")

        for run in tqdm(runs):
            try:
                if split == "val":
                    run_preds = run.val_preds
                elif split == "test_id":
                    run_preds = run.test_id_preds
                elif split == "test_ood":
                    run_preds = run.test_ood_preds
                elif split == "test_ood2":
                    run_preds = run.test_ood2_preds
                elif split == "test_ood4":
                    run_preds = run.test_ood4_preds
                else:
                    raise RuntimeError(f"Undefined split {split}")
                out_sm, label = extract_results(run_preds)

                # Remap split for nice reading
                print_split = {
                    "val": "validation",
                    "test_id": "ID",
                    "test_ood": "OOD",
                    "test_ood2": "OOD (center 2)",
                    "test_ood4": "OOD (center 4)",
                }[split]

                metric_dict = {}
                metric_dict["method"] = run.name
                # metric_dict["label_noise_lvl"] = run.label_noise_lvl

                if "Accuracy" in metrics:
                    acc = Accuracy()(out_sm, label)
                    metric_dict["Accuracy"] = acc.item()

                if "Balanced Accuracy" in metrics:
                    b_acc = Accuracy(average="macro", num_classes=2)(out_sm, label)
                    metric_dict["Balanced Accuracy"] = b_acc.item()

                if "F1" in metrics:
                    f1 = F1Score(num_classes=2, ignore_index=0)(out_sm, label)
                    metric_dict["F1"] = f1.item()

                if "AUROC" in metrics:
                    auroc = AUROC(pos_label=1, average="micro")(out_sm[:, 1], label)
                    metric_dict["AUROC"] = auroc.item()

                if "AUCPR" in metrics:
                    aucpr = AveragePrecision(pos_label=1, average="micro")(out_sm[:, 1], label)
                    metric_dict["AUCPR"] = aucpr.item()

                if "Precision" in metrics:
                    pre = Precision(num_classes=2, ignore_index=0)(out_sm, label)
                    metric_dict["Precision"] = pre.item()

                if "Recall" in metrics:
                    rec = Recall(num_classes=2, ignore_index=0)(out_sm, label)
                    metric_dict["Recall"] = rec.item()

                if "ECE" in metrics:
                    ece = CalibrationError(n_bins=20, norm="l1", compute_on_step=True)
                    ece.update(out_sm[:, 1], label)
                    ece = ece.compute()
                    metric_dict["ECE"] = ece.item()

                # if "ACE" in metrics:
                #     ace = AdaptiveCalibrationError()
                #     ace.update(out_sm[:, 1], label)
                #     ace = ace.compute()
                #     metric_dict["ACE"] = ace.item()

                results[(run.label_noise_lvl, str(run.exp_path))] = metric_dict
            except RuntimeError:
                print("Error in run", run.exp_path)

    df_id = pd.DataFrame.from_dict(results, orient="index")

    df_id = df_id.reset_index().rename(columns={"level_0": "split", "level_1": "run"})

    res_df = df_id.melt(id_vars=["run", "method", "split"], var_name="metric", value_name="value")

    sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    sns.set_style("whitegrid")
    sns.set_palette(color_palette)
    grid = sns.catplot(
        kind="box",
        x="split",
        y="value",
        hue="method",
        hue_order=[
            "ResNet",
            "ResNet Ensemble",
            "MCDO",
            "MCDO Ensemble",
            "TTA",
            "TTA Ensemble",
            "SVI",
            "SVI Ensemble",
        ],  # Had to be hardcoded, otherwise seaborn is sorting the SVI methods in the wrong order
        col="metric",
        col_wrap=3,
        data=res_df,
        # estimator=np.median,
        showfliers=False,
        sharey=False,
        saturation=1.0,
    )
    grid.figure.set_figwidth(15)
    grid.figure.set_figheight(5)
    grid.legend.set_title("Method")
    ax1 = grid.axes[0]
    ax1.set_title("Accuracy")
    ax1.set_xlabel(None)
    ax1.set_ylabel("Value")
    ax2 = grid.axes[1]
    ax2.set_title("Balanced Accuracy")
    ax2.set_xlabel(None)
    ax3 = grid.axes[2]
    ax3.set_title("ECE")
    ax3.set_xlabel(None)

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

    return df_id


def create_risk_reject_curve(
    results, metric_func, uc_thresh: str = "confidence", order: str = None, step_size: int = 1
):
    """Returns x,y values. x is reject-rate and sorted"""

    preds, labels = results["softmax"], results["label"]

    if uc_thresh == "confidence":
        x_val = compute_confidence(results["softmax"])
    elif uc_thresh == "entropy":
        x_val = normed_entropy(results["softmax"])
    elif uc_thresh == "variance":
        x_val = results["variance"][:, 1]
    else:
        raise RuntimeError("uc_thresh must be in [confidence, entropy, variance]")

    # x_val = x_val_func(results) if x_val_func is not None else preds
    if type(x_val) is torch.Tensor:
        x_val = x_val.numpy()

    if order is not None:

        ind = np.argsort(x_val)

        if order == "desc":
            ind = ind[::-1]

        ind = list(ind)
        preds = preds[ind]
        labels = labels[ind]
        x_val = x_val[ind]

    metric_val = []
    x_axis_val = []
    thresholds = []
    for i in range(0, len(x_val), step_size):

        if i > len(labels) - 1:
            continue
        else:
            ind = i
        # ind = min(len(labels) - 1, i)

        new_metric = metric_func(preds[ind:], labels[ind:])

        metric_val.append(new_metric)
        x_axis_val.append(ind)
        thresholds.append(x_val[ind])
    return x_axis_val, thresholds, metric_val


def compute_id_ood_reject_metrics(
    runs,
    metric_func=lambda preds, label: accuracy(preds, label, average="macro", num_classes=2),
    uc_thresh="confidence",
    order="asc",
    num_steps=1000,
    norm_x=True,
    x_range=None,
    rc_args: dict = None,
    color_palette="colorblind",
    fname: str = None,
):

    groups = {}
    auc = {}

    # Compute reject curve and save x and y in a dict of pandas series for id and ood data
    for i, run in enumerate(runs):

        # Skip Temp Scaling Runs
        if run.name == "Temp Scaling":
            continue

        for domain in ["test_id", "test_ood2", "test_ood4"]:
            results = run.get_preds(domain)

            assert isinstance(results, dict), "Results are expected to be of type dict"

            if uc_thresh == "variance" and "variance" not in results.keys():
                continue

            x, thres, met = create_risk_reject_curve(
                results,
                metric_func=metric_func,
                uc_thresh=uc_thresh,
                order=order,
                step_size=len(results["label"]) // num_steps,
            )

            if norm_x:
                x = np.array(x) / max(x)

            if not domain in groups.keys():
                groups[domain] = {}
                auc[domain] = {}

            if not run.name in groups[domain].keys():
                groups[domain][run.name] = {}
                auc[domain][run.name] = {}

            groups[domain][run.name][run.exp_path] = pd.Series(met, index=x, name=run.exp_path)
            auc[domain][run.name][run.exp_path] = torchmetrics.functional.auc(torch.tensor(x), torch.tensor((met)))

    methods = list(groups["test_id"].keys())

    for method in methods:

        # Put all experiments from one group into a dataframe (coloumn-wise).
        id_df = pd.DataFrame(groups["test_id"][method], dtype=float).interpolate()
        ood2_df = pd.DataFrame(groups["test_ood2"][method], dtype=float).interpolate()
        ood4_df = pd.DataFrame(groups["test_ood4"][method], dtype=float).interpolate()

        # If requested only plot a slice of the x_axis
        if x_range is not None:
            id_df = id_df.loc[x_range[0] : x_range[1]]
            ood2_df = ood2_df.loc[x_range[0] : x_range[1]]
            ood4_df = ood4_df.loc[x_range[0] : x_range[1]]

        id_df = id_df.reset_index().rename(columns={"index": "x"})
        ood2_df = ood2_df.reset_index().rename(columns={"index": "x"})
        ood4_df = ood4_df.reset_index().rename(columns={"index": "x"})

        # Convert dataframe from messy-wide-form into long-form (which is required by seaborn)
        id_df = id_df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")
        ood2_df = ood2_df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")
        ood4_df = ood4_df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")

        groups["test_id"][method] = id_df
        groups["test_ood2"][method] = ood2_df
        groups["test_ood4"][method] = ood4_df

    # Concatonate all groups. Introduce a column that identifies them by there group name.
    all_id = pd.concat(list(groups["test_id"].values()), keys=list(groups["test_id"].keys()), names=["method", "index"])
    all_ood2 = pd.concat(
        list(groups["test_ood2"].values()), keys=list(groups["test_ood2"].keys()), names=["method", "index"]
    )
    all_ood4 = pd.concat(
        list(groups["test_ood4"].values()), keys=list(groups["test_ood4"].keys()), names=["method", "index"]
    )

    # Stack those frames by id and ood.
    all_data = pd.concat([all_id, all_ood2, all_ood4], keys=["id", "ood2", "ood4"], names=["domain", "method", "index"])
    all_data = all_data.reset_index().drop(columns=["index"])
    # Invert x-Axis to show "Retained data"
    # all_data["x"] = 1 - all_data["x"]

    # Plot x vs. y differentiated by the method for the id domain and ood domain.
    sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    sns.set_style("whitegrid")
    # sns.set_palette(color_palette)

    if len(all_data["method"].unique()) == 8:
        # Use paired colors when all 8 methods are evaluated
        palette = sns.color_palette("Paired", 10)
        colors = [palette[1], palette[3], palette[5], palette[7]]
        colors = [color for color in colors for _ in (0, 1)]  # Duplicate colors
        grid = sns.relplot(
            x="x",
            y="value",
            col="domain",
            hue="method",
            style="method",
            kind="line",
            palette=colors,
            dashes=[(4, 2), ""] * 4,
            data=all_data,
            estimator=np.mean,
        )
    else:
        grid = sns.relplot(
            x="x",
            y="value",
            col="domain",
            hue="method",
            kind="line",
            palette=color_palette,
            data=all_data,
            estimator=np.mean,
        )
    grid.figure.set_figwidth(15)
    grid.figure.set_figheight(5)
    grid.legend.set_title("Method")
    ax1 = grid.axes[0, 0]
    ax1.set_title("ID centers", fontsize="medium")
    ax1.set_xlabel("Rejection rate")
    ax1.set_ylabel("Balanced Accuracy")
    ax2 = grid.axes[0, 1]
    ax2.set_title("OOD (center 2)", fontsize="medium")
    ax2.set_xlabel("Rejection rate")
    ax3 = grid.axes[0, 2]
    ax3.set_title("OOD (center 4)", fontsize="medium")
    ax3.set_xlabel("Rejection rate")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

    return auc


def visualize_tiles_by_uncertainty(
    run: Run, split: str, num: int = 20, nrows: int = 4, ncols: int = 5, fname: str = None
):

    assert num >= nrows * ncols, "num must be >= nrows * ncols"

    sns.set_context(context=None, font_scale=1, rc=RC_DICT)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    fig = plt.figure(constrained_layout=True, figsize=(15, 5))
    subfigs = fig.subfigures(1, 2, wspace=0.15)
    axsLeft = subfigs[0].subplots(nrows, ncols)
    axsRight = subfigs[1].subplots(nrows, ncols)

    preds, _, dataset = run.get_preds_and_dataset(split)
    preds = preds["softmax"]

    conf = compute_confidence(preds)

    conf_topk, conf_idx = torch.topk(conf, k=num, largest=True, sorted=True)
    unconf_topk, unconf_idx = torch.topk(conf, k=num, largest=False, sorted=True)
    # print(conf_topk)
    # print(unconf_topk)

    conf_tiles = torch.stack(
        [torchvision.transforms.functional.to_tensor(dataset.getImage(int(idx))) for idx in conf_idx]
    )
    conf_labels = [dataset.getlabel(int(idx)) for idx in conf_idx]
    unconf_tiles = torch.stack(
        [torchvision.transforms.functional.to_tensor(dataset.getImage(int(idx))) for idx in unconf_idx]
    )
    unconf_labels = [dataset.getlabel(int(idx)) for idx in unconf_idx]

    def __plot_batch(axs, tiles, labels):
        idx = 0
        for row in range(nrows):
            for col in range(ncols):
                img_tensor = tiles[idx]
                # img = (img - np.min(img)) / (np.max(img) - np.min(img))
                axs[row][col].imshow(img_tensor.permute([1, 2, 0]))
                axs[row][col].axis("off")
                axs[row][col].text(
                    0.0,
                    1.0,
                    labels[idx],
                    verticalalignment="top",
                    horizontalalignment="left",
                    fontsize="medium",
                    bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0),
                )
                idx += 1

    __plot_batch(axsLeft, conf_tiles, conf_labels)
    subfigs[0].suptitle("Tiles with lowest uncertainty", fontsize="medium")
    __plot_batch(axsRight, unconf_tiles, unconf_labels)
    subfigs[1].suptitle("Tiles with highest uncertainty", fontsize="medium")

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


def plot_slide_images(slide_images, idx: Union[int, list] = -1, save_path: Path = None):
    if idx == -1:
        indices = range(len(slide_images))
    elif type(idx) == int and idx < len(slide_images):
        indices = [idx]
    elif type(idx) == list and len(idx) < len(slide_images):
        indices = idx
    else:
        raise RuntimeError(f"Indices must be in [-1, {len(slide_images)}]")

    for i in indices:
        image_dict = slide_images[i]
        slidename = image_dict["slidename"]
        print("Slide: " + slidename)
        sns.set_context(context=None, font_scale=1, rc=RC_DICT)
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
        fig, axs = plt.subplots(1, 3, figsize=(15, 6))  # len(image_dict.keys()))
        legend_elements = [
            Patch(facecolor="magenta", edgecolor="k", label="Non-Tumor"),
            Patch(facecolor="cyan", edgecolor="k", label="Tumor"),
        ]
        for key in image_dict.keys():
            if key == "image":
                axs[0].set_title("Image", fontsize="medium")
                axs[0].imshow(image_dict[key])
                divider = make_axes_locatable(axs[0])
                # Hack to make subplots have the same height
                cax = divider.append_axes("right", size="5%", pad=0.05).set_visible(False)
                axs[0].axis("off")
            elif key == "class":
                axs[1].set_title("Annotation", fontsize="medium")
                axs[1].imshow(image_dict[key])
                axs[1].legend(handles=legend_elements, loc="lower right")
                divider = make_axes_locatable(axs[1])
                # Hack to make subplots have the same height
                cax = divider.append_axes("right", size="5%", pad=0.05).set_visible(False)
                axs[1].axis("off")
            elif key == "conf":
                axs[2].set_title("Tumor Confidence", fontsize="medium")
                background_img = image_dict["background"]
                conf_image = image_dict[key]
                # conf_masked = np.ma.masked_where(conf_image == 0, conf_image)
                im = axs[2].imshow(background_img, cmap=plt.cm.gray)
                im = axs[2].imshow(conf_image, cmap=plt.cm.viridis, clim=(0.0, 1.0))  # , vmin=0, vmax=1.0)
                divider = make_axes_locatable(axs[2])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
                axs[2].axis("off")
            else:
                pass
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(
                save_path / (slidename + ".pdf"), bbox_inches="tight"
            )  # Only bbox_inches manages to remove padding
        plt.show()


def compare_ece(
    runs,
    palette_name="tab10",
    color_insert_index: int = None,
    max_methods: int = 7,
    rc_args: dict = None,
    fname: str = None,
):
    sns.set_context(context=None, font_scale=1, rc=insert_rc_args(rc_args))
    sns.set_style("whitegrid")
    # sns.set_palette("colorblind")
    fig, ax = plt.subplots(figsize=(7, 5))
    ece = CalibrationError(n_bins=20, norm="l1")
    results = []

    for run in runs:
        try:
            results_id = run.test_id_preds
            results_ood2 = run.test_ood2_preds
            results_ood4 = run.test_ood4_preds

            out_sm_id, label_id = extract_results(results_id)
            out_sm_ood2, label_ood2 = extract_results(results_ood2)
            out_sm_ood4, label_ood4 = extract_results(results_ood4)

            ece.update(out_sm_id[:, 1], label_id)
            ece_id = ece.compute().item()
            ece.reset()
            ece.update(out_sm_ood2[:, 1], label_ood2)
            ece_ood2 = ece.compute().item()
            ece.reset()
            ece.update(out_sm_ood4[:, 1], label_ood4)
            ece_ood4 = ece.compute().item()
            ece.reset()
            id_entry = {
                "method": run.name,  # run_name_lookup[run.name.split("/")[-3]],
                "distrib": "ID centers",
                "ece": ece_id,
            }
            ood2_entry = {
                "method": run.name,  # run_name_lookup[run.name.split("/")[-3]],
                "distrib": "OOD (center 2)",
                "ece": ece_ood2,
            }
            ood4_entry = {
                "method": run.name,  # run_name_lookup[run.name.split("/")[-3]],
                "distrib": "OOD (center 4)",
                "ece": ece_ood4,
            }
            results.extend([id_entry, ood2_entry, ood4_entry])
        except RuntimeError:
            print("Error in run", run.exp_path)

    df = pd.DataFrame(results)
    n_methods = len(df["method"].unique())

    if color_insert_index is not None:
        palette = sns.color_palette(palette_name, max_methods)
        palette.insert(color_insert_index, palette[-1])
        palette.pop(-1)
    else:
        palette = palette_name

    ax = sns.boxplot(  # Use boxplot in case of multiple runs per method
        data=df,
        x="distrib",
        y="ece",
        hue="method",
        fliersize=0,  # Do not show outliers
        # linewidth=0,  # Changes the linewidth for all boxes, not customizable
        palette=palette,
    )
    ax.set_xlabel(None)
    # Filter out box patches from all available patches
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    # Remove frame lines from first group of boxes
    for i in range(0, n_methods):
        col = box_patches[i].get_facecolor()
        box_patches[i].set_edgecolor(col)
        # box_patches[i]._linewidth = 0
    # Remove whiskers and median line from first group of boxes (each box has 6 lines)
    for i in range(0, n_methods * 6):
        ax.lines[i]._linewidth = 0
    ax.set_ylabel("ECE")
    ax.legend(title="Method", loc="upper left")
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    user = "Alex"

    if user == "Hendrik":
        CONFIG_PATH = "logs/config.yaml"
        RUN_DIR = os.environ["EXPERIMENT_LOCATION"]
        runs = [
            *create_runs_from_folder(
                "shared/Baseline_ResNet/", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="ResNet"
            ),
            *create_runs_from_folder("shared/BaseLine_DE/", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="Ensemble"),
            *create_runs_from_folder("shared/BaseLine_MCDO/", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="MCDO"),
            *create_runs_from_folder("shared/BaseLine_TTA/", run_dir=RUN_DIR, config_path=CONFIG_PATH, name="TTA"),
        ]

        compute_id_ood_reject_metrics(runs, num_steps=20, x_range=[0.1, 0.8])
        # compute_id_ood_auroc(runs)
        # compute_metrics(runs, "test_id")
        compare_ece(runs)

        exit()
    elif user == "Alex":
        CONFIG_PATH = "logs/config.yaml"
        RUN_DIR = Path("/mnt/smb/OE0601-Projekte/KTI/UncerHisto/Cam17_final")
        runs = [
            *create_runs_from_folder(
                Path("strong/resnet"),
                name="ResNet",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("strong/resnet_ensemble"),
                name="ResNet Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
            ),
            *create_runs_from_folder(
                Path("strong/mcdo"),
                name="MCDO",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("strong/mcdo_ensemble"),
                name="MCDO Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
            ),
            *create_runs_from_folder(
                Path("strong/tta"),
                name="TTA",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("strong/tta_ensemble"),
                name="TTA Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
            ),
            *create_runs_from_folder(
                Path("strong/svi"),
                name="SVI",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                max_num=5,
                properties={"label_noise_lvl": "25%"},
            ),
            *create_ensemble_from_folder(
                Path("strong/svi"),
                name="SVI Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "25%"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/0-Threshold/LNResNetNone"),
                name="ResNet",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/0-Threshold/LNResNetNone"),
                name="ResNet Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseMCDONone"),
                name="MCDO",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseMCDONone"),
                name="MCDO Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseTTANone"),
                name="TTA",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseTTANone"),
                name="TTA Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseSVINone"),
                name="SVI",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/0-Threshold/LabelNoiseSVINone"),
                name="SVI Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "0%"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseResNetUniform"),
                name="ResNet",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseResNetUniform"),
                name="ResNet Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseMCDOUniform"),
                name="MCDO",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseMCDOUniform"),
                name="MCDO Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseTTAUniform"),
                name="TTA",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseTTAUniform"),
                name="TTA Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseSVIUniform"),
                name="SVI",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Uniform/LabelNoiseSVIUniform"),
                name="SVI Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Uniform"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Edge/LabelNoiseResNetEdgeUniform"),
                name="ResNet",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Edge/LabelNoiseResNetEdgeUniform"),
                name="ResNet Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Edge/LabelNoiseMCDOEdgeUniform"),
                name="MCDO",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Edge/LabelNoiseMCDOEdgeUniform"),
                name="MCDO Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Edge/LabelNoiseTTAEdgeUniform"),
                name="TTA",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Edge/LabelNoiseTTAEdgeUniform"),
                name="TTA Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
            ),
            *create_runs_from_folder(
                Path("LabelNoise/Edge/LabelNoiseSVIEdgeUniform"),
                name="SVI",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
                max_num=5,
            ),
            *create_ensemble_from_folder(
                Path("LabelNoise/Edge/LabelNoiseSVIEdgeUniform"),
                name="SVI Ensemble",
                run_dir=RUN_DIR,
                config_path=CONFIG_PATH,
                properties={"label_noise_lvl": "Border"},
            ),
        ]
        print("Total number of runs: ", len(runs))
        compute_metrics_label_noise(
            runs,
            split=["test_id"],
            metrics=["Accuracy", "Balanced Accuracy", "ECE"],
            rc_args={"xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "legend.title_fontsize": 12},
            color_palette="Paired",
            # fname=SAVE_PATH / "label_noise_boxplot.pdf",
        )
        exit()
    else:
        version = 3
        runs0 = [
            Run("cluster/Cam17_v2/resnet/version_0"),
            Run("cluster/Cam17_v2/mcdo-01/version_0"),
            Run("cluster/Cam17_v2/ensemble/version_0"),
            # Run("cluster/Cam17_v2/ts/version_0"),
            Run("cluster/Cam17_v2/tta/version_0"),
            Run("cluster/Cam17_v2/svi/version_0"),
        ]
        runs1 = [
            Run("cluster/Cam17_v2/resnet/version_1"),
            Run("cluster/Cam17_v2/mcdo-01/version_1"),
            Run("cluster/Cam17_v2/ensemble/version_1"),
            # Run("cluster/Cam17_v2/ts/version_1"),
            Run("cluster/Cam17_v2/tta/version_1"),
            Run("cluster/Cam17_v2/svi/version_1"),
        ]
        runs2 = [
            Run("cluster/Cam17_v2/resnet/version_2"),
            Run("cluster/Cam17_v2/mcdo-01/version_2"),
            Run("cluster/Cam17_v2/ensemble/version_2"),
            # Run("cluster/Cam17_v2/ts/version_2"),
            Run("cluster/Cam17_v2/tta/version_2"),
            Run("cluster/Cam17_v2/svi/version_2"),
        ]

        # compute_id_ood_auroc(runs)
        # compute_id_ood_reject_metrics(runs)
        compare_ece(runs2, version=version, save=False)

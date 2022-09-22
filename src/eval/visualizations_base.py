import os
import sys
import warnings
from typing import Union, Optional, Sequence

from argparse import ArgumentParser
from pathlib import Path
import json
import yaml
import itertools
import functools
from tqdm import tqdm

import torch
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
from torchmetrics.functional import accuracy, roc
import torchvision

import numpy as np
import matplotlib

# matplotlib.use("Agg")  # Fixed issue when script was stuck at import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import seaborn as sns

from ..metrics.metrics import compute_confidence, normed_entropy
from ..datamodules import camelyon_datamodules

warnings.simplefilter("ignore", category=FutureWarning, lineno=0, append=False)


class Run:
    def __init__(
        self,
        exp_dir: str,
        checkpoint_name="final",
        run_dir: str = os.environ["EXPERIMENT_LOCATION"],
        pred_path: str = "predictions",
        config_path: str = "logs/config.yaml",
        name: Union[str, None] = None,
        lazy: bool = True,
        properties: Optional[dict] = None,
    ):

        exp_path = Path(run_dir) / exp_dir
        config_path = exp_path / config_path

        pred_file_names = {
            "val": f"test_val_{checkpoint_name}.preds",
            "test_id": f"test_id_{checkpoint_name}.preds",
            "test_ood": f"test_ood_{checkpoint_name}.preds",
            "test_ood2": f"test_ood2_{checkpoint_name}.preds",
            "test_ood4": f"test_ood4_{checkpoint_name}.preds",
        }

        preds_path = {key: exp_path / pred_path / value for key, value in pred_file_names.items()}

        self.exp_path = exp_path
        self.pred_path = exp_path / pred_path
        assert exp_path.exists()
        assert config_path.exists()

        if properties is not None:
            self.properties = properties
            for key, value in properties.items():
                setattr(self, key, value)

        # for _, path in preds_path.items():
        #    assert path.exists(), f"Path {str(path)} does not exists"

        self._config = None
        self._preds_path = preds_path
        self._preds = {key: None for key in preds_path}

        self.config_file = config_path

        self.name = exp_dir + "/" + checkpoint_name if name is None else name

        if not lazy:
            self.val_preds
            self.test_id_preds
            self.test_ood_preds

    @property
    def val_preds(self):

        if self._preds["val"] is None:
            self._preds["val"] = torch.load(str(self._preds_path["val"]))

        return self._preds["val"]

    @property
    def test_id_preds(self):

        if self._preds["test_id"] is None:
            self._preds["test_id"] = torch.load(str(self._preds_path["test_id"]))

        return self._preds["test_id"]

    @property
    def test_ood_preds(self):

        if self._preds["test_ood"] is None:
            self._preds["test_ood"] = torch.load(str(self._preds_path["test_ood"]))

        return self._preds["test_ood"]

    @property
    def test_ood2_preds(self):

        if self._preds["test_ood2"] is None:
            self._preds["test_ood2"] = torch.load(str(self._preds_path["test_ood2"]))

        return self._preds["test_ood2"]

    @property
    def test_ood4_preds(self):

        if self._preds["test_ood4"] is None:
            self._preds["test_ood4"] = torch.load(str(self._preds_path["test_ood4"]))

        return self._preds["test_ood4"]

    @property
    def config(self):

        if self._config is None:
            self._config = yaml.safe_load(open(str(self.config_file), "r"))

        return self._config

    def get_preds(self, split: str):
        if split == "test_id":
            return self.test_id_preds
        elif split == "test_ood":
            return self.test_ood_preds
        elif split == "test_ood2":
            return self.test_ood2_preds
        elif split == "test_ood4":
            return self.test_ood4_preds
        else:
            path = self.pred_path / (split + ".preds")
            if path.exists():
                preds = torch.load(str(path))
                return preds
            else:
                RuntimeError("Split must be in [test_id, test_ood, test_ood2, test_ood4] or {str(path)'} must exist!")

    def get_preds_and_dataset(self, split: str):
        assert split in ["train", "test_id", "test_ood", "test_ood2", "test_ood4"]

        data_config = self.config["data"]

        # get correct class
        data_class = getattr(globals()["camelyon_datamodules"], self.config["data_class"])

        # Replace paths if trained on cluster or workstation.
        general_data_path = Path(os.environ["DATASET_LOCATION"])
        if "path_Cam17" in data_config:
            data_config["path"] = general_data_path / "Camelyon16/tiles"
            data_config["path_Cam17"] = general_data_path / "Camelyon17/tiles"
        else:
            data_config["path"] = general_data_path / "Camelyon17/tiles"

        data_module = data_class(**data_config)

        # Get the correct dataset and predictions
        if split == "train":
            raise RuntimeError("To lazy to guarantee correct values for train.")
        elif split == "val":
            data_module.setup("val")
            dataset = data_module.val_dataloader().dataset
            preds = self.val_preds
        elif split in ["test", "test_id", "test_ood"]:
            data_module.setup("test")
            test_id, test_ood = data_module.test_dataloader()

            if split == "test_ood":
                dataset = test_ood.dataset
                preds = self.test_ood_preds
            else:
                dataset = test_id.dataset
                preds = self.test_id_preds
        elif split == "test_ood2":
            data_config["ood_centers"] = 2
            data_module = data_class(**data_config)
            data_module.setup("test")
            _, test_ood = data_module.test_dataloader()
            dataset = test_ood.dataset
            preds = self.test_ood2_preds
        elif split == "test_ood4":
            data_config["ood_centers"] = 4
            data_module = data_class(**data_config)
            data_module.setup("test")
            _, test_ood = data_module.test_dataloader()
            dataset = test_ood.dataset
            preds = self.test_ood4_preds

        return preds, data_module, dataset


class EnsembleRun(Run):
    def __init__(
        self,
        runs: Sequence[Run],
        name: str,
        lazy: bool = True,
    ):

        self.runs = runs

        # exp_path = Path(run_dir) / exp_dir
        # config_path = exp_path / config_path
        #
        # pred_file_names = {
        #     "val": f"val_{checkpoint_name}.preds",
        #     "test_id": f"test_{checkpoint_name}.preds",
        #     "test_ood": f"test_ood_{checkpoint_name}.preds",
        # }
        #
        # preds_path = {key: exp_path / pred_path / value for key, value in pred_file_names.items()}

        self.exp_path = runs[0].exp_path

        for run in runs:
            if hasattr(run, "properties"):
                for key, value in run.properties.items():
                    setattr(self, key, value)

        # assert exp_path.exists()
        # assert config_path.exists()
        #
        # for _, path in preds_path.items():
        #     assert path.exists(), f"Path {str(path)} does not exists"

        self._config = None
        self._preds = {key: None for key in ["val", "test_id", "test_ood", "test_ood2", "test_ood4"]}

        self.config_file = runs[0].config_file  # config_path

        self.name = name

        if not lazy:
            self.val_preds
            self.test_id_preds
            self.test_ood_preds

    def _compute_ensemble_preds(self, preds: Sequence[dict[torch.Tensor]]):

        new_keys = ["softmax", "label"]
        for key in new_keys:
            assert key in preds[0].keys()

        softmaxes = torch.stack([pred["softmax"] for pred in preds])

        var, mean = torch.var_mean(softmaxes, dim=0)

        # If this is true, we create an ensemble out of ensembles.
        # Use the variance formula for mixed distributions!
        # https://en.wikipedia.org/wiki/Mixture_distribution#Moments
        # We further assume that all subpopulations are of equal size! This is important.
        if "variance" in preds[0].keys():

            # Tricks used here only work for binary classification.
            assert softmaxes.shape[2] == 2

            # Hardcoded: Undo Bessels correction.
            # To bad we dont safe the number of iterations oh well.
            variances = 9 / 10 * torch.stack([pred["variance"] for pred in preds])

            # Use forumula again with bessels correction (redo Bessels).
            # https://stats.stackexchange.com/questions/289256/how-to-compute-mean-and-standard-deviation-of-a-set-of-values-each-of-which-is-i
            new_var = torch.sum(10 * (softmaxes - mean.unsqueeze(0)) ** 2 + variances, dim=0)
            new_var /= (10 * len(preds)) - 1

        label = preds[0]["label"]
        # for pred in preds:
        #     assert (label == pred["label"]).all()

        return {"softmax": mean, "label": label, "variance": var}

    @property
    def val_preds(self):

        if self._preds["val"] is None:
            preds = [run.val_preds for run in self.runs]

            self._preds["val"] = self._compute_ensemble_preds(preds)

        return self._preds["val"]

    @property
    def test_id_preds(self):

        if self._preds["test_id"] is None:
            preds = [run.test_id_preds for run in self.runs]

            self._preds["test_id"] = self._compute_ensemble_preds(preds)
        return self._preds["test_id"]

    @property
    def test_ood_preds(self):

        if self._preds["test_ood"] is None:
            preds = [run.test_ood_preds for run in self.runs]

            self._preds["test_ood"] = self._compute_ensemble_preds(preds)

        return self._preds["test_ood"]

    @property
    def test_ood2_preds(self):

        if self._preds["test_ood2"] is None:
            preds = [run.test_ood2_preds for run in self.runs]

            self._preds["test_ood2"] = self._compute_ensemble_preds(preds)

        return self._preds["test_ood2"]

    @property
    def test_ood4_preds(self):

        if self._preds["test_ood4"] is None:
            preds = [run.test_ood4_preds for run in self.runs]

            self._preds["test_ood4"] = self._compute_ensemble_preds(preds)

        return self._preds["test_ood4"]

    @property
    def config(self):

        if self._config is None:
            self._config = yaml.safe_load(open(str(self.config_file), "r"))

        return self._config

    def get_preds(self, split: str):
        if split == "test_id":
            return self.test_id_preds
        elif split == "test_ood":
            return self.test_ood_preds
        elif split == "test_ood2":
            return self.test_ood2_preds
        elif split == "test_ood4":
            return self.test_ood4_preds
        else:
            preds = [run.get_preds(split) for run in self.runs]
            preds = self._compute_ensemble_preds(preds)
            return preds


def create_runs_from_folder(
    exp_dir: Path,
    checkpoint: str = "final",
    name: Optional[str] = None,
    max_num: Optional[int] = None,
    run_dir=os.environ["EXPERIMENT_LOCATION"],
    config_path: str = "logs/config.yaml",
    prediction_path="predictions",
    properties=None,
):

    runs = []

    dir = Path(run_dir) / Path(exp_dir)

    if not dir.exists():
        raise RuntimeError(f"Provided path {dir} does not exist.")

    configs = sorted(list(dir.glob(f"**/{config_path}")))

    if max_num is not None:
        configs = configs[-max_num:]

    if len(configs) == 0:
        warnings.warn(f"Found no runs in {dir}.")

    for config in configs:
        config = Path(config)

        assert config.parts[-2] == "logs"

        tmp_dir = config.parents[1].relative_to(run_dir)

        runs.append(
            Run(
                tmp_dir,
                checkpoint_name=checkpoint,
                pred_path=prediction_path,
                config_path=config_path,
                run_dir=run_dir,
                name=name,
                properties=properties,
            )
        )

    return runs


def create_ensemble_from_folder(
    exp_dir: Path,
    members_per_ensemble: int = 5,
    checkpoint: str = "final",
    name: Optional[str] = None,
    run_dir=os.environ["EXPERIMENT_LOCATION"],
    config_path: str = "logs/config.yaml",
    prediction_path="predictions",
    properties=None,
):

    runs = create_runs_from_folder(
        exp_dir, checkpoint, name, None, run_dir, config_path, prediction_path, properties=properties
    )

    runs = sorted(runs, key=lambda x: str(x.exp_path))
    # assert len(runs) % members_per_ensemble == 0
    num_ens = len(runs) // members_per_ensemble
    left_over = len(runs) % members_per_ensemble
    runs = runs[: num_ens * members_per_ensemble]  # Only keep number of runs that are multiples of members_per_ensemble

    enss = np.array_split(np.array(runs), num_ens)

    returns = []
    for ens in enss:
        if len(ens) == members_per_ensemble:
            returns.append(EnsembleRun(ens, name))

    return returns


def tile_extractor(preds):

    out = torch.cat(list(map(lambda x: torch.stack(x[0]), preds)), dim=1)
    sm = torch.cat(list(map(lambda x: x[1], preds)), dim=0)
    label = torch.cat(list(map(lambda x: x[2], preds)), dim=0)

    return out, sm, label


def extract_results(results):

    if isinstance(results, list):
        if len(results) == 2:
            out_sm, y = results
        elif len(results) == 3:
            out_sm, y, var_sm = results
        else:
            print("Unknown length of results")
            return -1
        return out_sm, y

    if isinstance(results, dict):
        return results["softmax"], results["label"]


def extract_var(results):
    """
    var_sm includes variance for both classes,
    but both variances are the same, since tumor_prob = 1 - non_tumor_prob
    """
    out_sm, y, var_sm = results
    return var_sm[:, 1]  # Only return tumor variance


def generate_slide_vis_from_run(run: Run, split: str, select_slides=None, tile_render_size: int = 8):

    preds, _, dataset = run.get_preds_and_dataset(split)
    preds = preds["softmax"]
    images = dataset.build_slide_lvl_images(
        pred_list=preds, tile_render_size=tile_render_size, select_slides=select_slides
    )

    return images


def generate_slide_preds_from_run(
    run: Run, split: str, select_slides=None, tumor_thresh: float = 0.5, approach: str = "convolution"
):

    preds, _, dataset = run.get_preds_and_dataset(split)
    preds = preds["softmax"]
    slide_preds = dataset.generate_slide_prediction(
        pred_list=preds,
        select_slides=select_slides,
        tumor_thresh=tumor_thresh,
        approach=approach,
    )

    return slide_preds


def visualize_slides(slides, combine: bool = False, save_path=None):
    slide_idx = 0
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

    if combine:
        fig, all_axes = matplotlib.pyplot.subplots(len(slides), len(slides[0].keys()) - 1)

    for i, image_dict in enumerate(slides):

        name = image_dict["slidename"]
        keys = list(image_dict.keys())
        keys.remove("slidename")

        if not combine:
            fig, axes = matplotlib.pyplot.subplots(1, len(keys))
        else:
            axes = all_axes[i, :]

        for j, key in enumerate(keys):

            if not combine or i == 0:
                axes[j].set_title(str(key))
            if combine and j == 0:
                axes[j].set_ylabel(str(name))
            axes[j].imshow(image_dict[key])
            axes[j].axis("off")

        # fig.tight_layout()
        # fig.suptitle(str(name))
        print(f"({slide_idx})" + str(name))

        if not combine:
            if save_path is not None:
                fig.savefig((save_path / f"{slide_idx}.svg").absolute(), dpi=1000)
            matplotlib.pyplot.show()
            matplotlib.pyplot.close()
            slide_idx += 1

    if combine:
        matplotlib.pyplot.show()
        if save_path:
            fig.savefig((save_path / "slides.svg").absolute(), dpi=1000)


def get_tiles_by_uncertainty(runs: Union[Run, list[Run]], split: str, num: int = 20):

    if not isinstance(runs, list):
        runs = [runs]

    conf = 0.0

    for run in runs:

        preds, _, dataset = run.get_preds_and_dataset(split)
        preds = preds["softmax"]

        conf = compute_confidence(preds) + conf

    conf /= len(runs)

    _, conf_idx = torch.topk(conf, k=num, largest=True, sorted=True)
    _, unconf_idx = torch.topk(conf, k=num, largest=False, sorted=True)

    conf_tiles = torch.stack(
        [torchvision.transforms.functional.to_tensor(dataset.getImage(int(idx))) for idx in conf_idx]
    )
    unconf_tiles = torch.stack(
        [torchvision.transforms.functional.to_tensor(dataset.getImage(int(idx))) for idx in unconf_idx]
    )

    conf_grid = torchvision.utils.make_grid(conf_tiles, n_row=5).permute([2, 1, 0])

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(conf_grid)

    unconf_grid = torchvision.utils.make_grid(unconf_tiles, n_row=5).permute([2, 1, 0])
    axes[1].imshow(unconf_grid)
    plt.show()


def average_disagreement(preds):
    dis = 0.0
    c = 0.0
    for i in range(len(preds)):
        for j in range(len(preds)):

            if i >= j:
                continue
            c += 1

            class_i = torch.argmax(preds[i], dim=1)
            class_j = torch.argmax(preds[j], dim=1)

            dis += (class_i != class_j).sum()

    dis /= c
    dis /= len(preds[0])
    return dis


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
):

    if isinstance(split, str):
        splits = [split]
    else:
        splits = split

    results = {}

    for split in splits:

        print(f"Computing {split} metrics.")

        for run in tqdm(runs):

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

    res_df = pd.DataFrame.from_dict(results, orient="index")

    # if drop_metrics is not None:
    #     res_df = res_df.drop(columns=drop_metrics)

    res_df = res_df.reset_index().rename(columns={"level_0": "split", "level_1": "run"})

    return res_df
    # res_df.drop(columns="split")

    res_df = res_df.melt(id_vars=["split", "run", "method"], var_name="metric", value_name="value")

    fig = sns.catplot(
        kind="box",
        x="split",
        y="value",
        hue="method",
        col="metric",
        col_wrap=3,
        data=res_df,
        # estimator=np.median,
        showfliers=False,
        sharey=False,
        legend_out=False,
    )

    # melt_df = res_df.melt(id_vars=["index", "method"], var_name="metric", value_name="value")
    # sns.catplot(x="method", y="value", col="metric", kind="box", data=melt_df, col_wrap=3)
    # plt.show()

    grouped_results = res_df.groupby(by=["split", "method"])

    mean_var = grouped_results.agg([np.mean, np.std]).T

    print(mean_var)

    if compute_significance:
        raise RuntimeError("Computing significance not supported in this commit")
        print("Computing significance")
        methods = res_df["method"].unique()
        number_runs_per_method = res_df["method"].value_counts()

        import scipy.stats as scs

        for metric in metrics:

            p_matrix = pd.DataFrame(np.ones((len(methods), len(methods))), index=methods, columns=methods)

            for method in methods:
                for method2 in methods:
                    if method != method2:

                        t, p = scs.ttest_ind(
                            res_df[metric][res_df["method"] == method].to_numpy(),
                            res_df[metric][res_df["method"] == method].to_numpy(),
                            equal_var=False,
                            alternative="greater",
                            permutations=1000,
                        )
                        p_matrix.at[method, method2] = p

            print(metric)
            print(p_matrix)

    plt.tight_layout()
    plt.show()

    return fig


def clip_range(array, target_arrays, low, high):
    # Find lower and upper bounder
    import bisect

    low_pos = bisect.bisect_left(array, low)
    high_pos = bisect.bisect_right(array, high)

    array = array[low_pos:high_pos]

    for i in range(len(target_arrays)):
        target_arrays[i] = target_arrays[i][low_pos:high_pos]

    return array, target_arrays


def combine_group(group, average=True, desc=False):
    # For now ignore x

    merged_group_dict = {}

    group_xs = [np.array(group_run["x"]) for group_run in group]
    group_ys = [np.array(group_run["y"]) for group_run in group]

    union_x = functools.reduce(np.union1d, group_xs)
    group_ys = [torch.tensor(np.interp(union_x, g_x, g_y)) for g_x, g_y in zip(group_xs, group_ys)]

    if average:
        if desc:
            union_x = union_x[::-1]

        if len(group_ys) > 1:
            var_y, m_y = torch.std_mean(torch.stack(group_ys, dim=0), dim=0)
        else:
            m_y = group_ys[0]
            var_y = torch.zeros_like(m_y)
        merged_group_dict["m_y"] = m_y
        merged_group_dict["var_y"] = var_y
        merged_group_dict["x"] = union_x
        merged_group_dict["y"] = group_ys

    else:
        merged_group_dict["y"] = group_ys
        merged_group_dict["x"] = union_x

    return merged_group_dict


def create_risk_reject_curve(preds, labels, metric_func, x_val_func, order: str = None, step_size: int = 1):
    """Returns x,y values. x is reject-rate and sorted"""
    x_val = x_val_func(preds, labels) if x_val_func is not None else preds

    if order is not None:

        ind = np.argsort(x_val)

        if order == "desc":
            ind = ind[::-1]

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
    x_val_func=lambda preds, label: compute_confidence(preds),
    order="asc",
    num_steps=1000,
    norm_x=True,
    x_range=None,
):

    id_groups = {}
    ood_groups = {}

    # Compute reject curve and save x and y in a dict of pandas series for id and ood data
    for i, run in enumerate(runs):

        results_id = run.test_id_preds
        results_ood = run.test_ood_preds

        out_sm_id, label_id = extract_results(results_id)
        out_sm_ood, label_ood = extract_results(results_ood)

        step_size = len(label_id) // num_steps

        x, thres, met = create_risk_reject_curve(
            out_sm_id, label_id, metric_func=metric_func, x_val_func=x_val_func, order=order, step_size=step_size
        )

        if norm_x:
            x = np.array(x) / max(x)

        if not run.name in id_groups.keys():
            id_groups[run.name] = {}

        id_groups[run.name][run.exp_path] = pd.Series(met, index=x, name=run.exp_path)

        x, thres, met = create_risk_reject_curve(
            out_sm_ood, label_ood, metric_func=metric_func, x_val_func=x_val_func, order=order, step_size=step_size
        )

        if norm_x:
            x = np.array(x) / max(x)

        if not run.name in ood_groups.keys():
            ood_groups[run.name] = {}

        ood_groups[run.name][run.exp_path] = pd.Series(met, index=x, name=run.exp_path)

    groups = list(id_groups.keys())

    for group in groups:

        # Put all experiments from one group into a dataframe (coloumn-wise).
        id_df = pd.DataFrame(id_groups[group], dtype=float).interpolate()
        ood_df = pd.DataFrame(ood_groups[group], dtype=float).interpolate()

        # If requested only plot a slice of the x_axis
        if x_range is not None:
            id_df = id_df.loc[x_range[0] : x_range[1]]
            ood_df = ood_df.loc[x_range[0] : x_range[1]]

        id_df = id_df.reset_index().rename(columns={"index": "x"})
        ood_df = ood_df.reset_index().rename(columns={"index": "x"})

        # Convert dataframe from messy-wide-form into long-form (which is required by seaborn)
        id_df = id_df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")
        ood_df = ood_df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")

        id_groups[group] = id_df
        ood_groups[group] = ood_df

    # Concatonate all groups. Introduce a column that identifies them by there group name.
    all_id = pd.concat(list(id_groups.values()), keys=list(id_groups.keys()), names=["method", "index"])
    all_ood = pd.concat(list(ood_groups.values()), keys=list(ood_groups.keys()), names=["method", "index"])

    # Stack those frames by id and ood.
    all_data = pd.concat([all_id, all_ood], keys=["id", "ood"], names=["domain", "method", "index"])
    all_data = all_data.reset_index().drop(columns=["index"])

    # Plot x vs. y differentiated by the method for the id domain and ood domain.
    fig = sns.relplot(
        x="x",
        y="value",
        hue="method",
        col="domain",
        kind="line",
        data=all_data,
        facet_kws={"sharey": False, "sharex": True},
    )

    return fig


def compute_id_ood_auroc2(runs, plot_std=False, colors=None):
    c_map = plt.cm.get_cmap("Dark2")

    fig, ax = plt.subplots()

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    result_dict = {}

    for run in runs:

        results_id = run.test_id_preds
        results_ood = run.test_ood_preds

        out_sm_id, _ = extract_results(results_id)
        out_sm_ood, _ = extract_results(results_ood)

        label_id = torch.ones(len(out_sm_id))
        label_ood = torch.zeros(len(out_sm_ood))

        labels = torch.cat([label_id, label_ood])
        preds = torch.cat([torch.Tensor(out_sm_id), torch.Tensor(out_sm_ood)])

        conf = torch.max(preds, dim=-1)[0]
        conf = torch.stack([1 - conf, conf], dim=-1)

        fpr, tpr, thres = roc(conf, labels, num_classes=2)

        result_dict[str(run.exp_path)] = (fpr, tpr, thres)

    groups = {}

    for run in runs:
        fpr, tpr, thres = result_dict[str(run.exp_path)]
        if not run.name in groups.keys():
            groups[run.name] = []
        groups[run.name].append({"x": fpr[1], "y": tpr[1]})
    for i, group in enumerate(groups.keys()):
        groups[group] = combine_group(groups[group], average=True)
        group_results = groups[group]
        x = group_results["x"]
        m_y = group_results["m_y"]
        var_y = group_results["var_y"]
        ax.plot(x, m_y, label=group, color=colors[i] if colors is not None else c_map(i))
        if plot_std:
            ax.fill_between(x, m_y + var_y, m_y - var_y, color=colors[i] if colors is not None else c_map(i), alpha=0.1)
    ax.legend()
    plt.show()


def compute_id_ood_auroc(runs: Sequence[Run]) -> None:

    groups = {}

    # Compute ROC curve and save x and y in a dict of pandas series
    for run in runs:

        results_id = run.test_id_preds
        results_ood = run.test_ood_preds

        out_sm_id, _ = extract_results(results_id)
        out_sm_ood, _ = extract_results(results_ood)

        label_id = torch.ones(len(out_sm_id))
        label_ood = torch.zeros(len(out_sm_ood))

        labels = torch.cat([label_id, label_ood])
        preds = torch.cat([torch.Tensor(out_sm_id), torch.Tensor(out_sm_ood)])

        conf = torch.max(preds, dim=-1)[0]
        conf = torch.stack([1 - conf, conf], dim=-1)

        fpr, tpr, thres = roc(conf, labels, num_classes=2)

        if not run.name in groups.keys():
            groups[run.name] = {}

        # We need to clean the ROC of duplicate x values...
        s = pd.Series(tpr[1].numpy(), index=fpr[1].numpy())
        s = s.iloc[~s.index.duplicated(keep="last")]

        groups[run.name][str(run.exp_path)] = s

    group_names = list(groups.keys())

    for group in group_names:

        # Put all experiments from one group into a dataframe (coloumn-wise). As they have differing x-axis, interpolate the missing values.
        df = pd.DataFrame(groups[group], dtype=float).interpolate().reset_index().rename(columns={"index": "x"})

        # Convert dataframe from messy-wide-form into long-form (which is required by seaborn)
        df = df.melt(id_vars="x", var_name="run", value_name="value").drop(columns="run")

        groups[group] = df

    # Concatonate all groups. Introduce a column that identifies them by there group name.
    all_data = pd.concat(list(groups.values()), keys=list(groups.keys()), names=["method", "index"])
    all_data = all_data.reset_index().drop(columns=["index"])

    # Plot x vs. y, differentiated by the method. Use the standard deviation as confidence interval, as bootstraping takes way to long.
    sns.relplot(x="x", y="value", hue="method", kind="line", data=all_data, ci="sd")
    plt.show()


if __name__ == "__main__":

    user = "Hendrik"

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
        compute_id_ood_auroc(runs)

        exit()
    else:
        version = 2
        runs = [
            Run("resnet/version_0"),
            Run("ts/version_0"),
            Run("tta/version_1"),
            Run("ensemble/version_0"),
            Run("mcdo/version_2"),
        ]

        # compute_id_ood_auroc(runs)

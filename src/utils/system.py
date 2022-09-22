import os
from pathlib import Path
from typing import Union


RC_DICT = {
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 14,
}


def insert_rc_args(rc_args: dict):
    rc_dict = RC_DICT.copy()
    if rc_args is not None:
        for key in rc_args:
            if key in rc_dict.keys():
                rc_dict[key] = rc_args[key]
            else:
                print(key + " is not a valid argument")
    return rc_dict


def find_max_version(path: Union[Path, str]):
    path = Path(path)

    version_dirs = list(path.glob("version_*"))

    if len(version_dirs) == 0:
        return -1

    version_dirs = [entry for entry in version_dirs if entry.is_dir()]

    versions = [int(str(dir_name).split("_")[-1]) for dir_name in version_dirs]
    max_version = max(versions)
    return max_version


def find_save(path, last=False, by_metric="epoch", mode_max=True):
    cktp_files = list(path.glob("*.ckpt"))

    if len(cktp_files) == 0:
        return None

    if last:
        for file in cktp_files:
            if file.stem == "last":
                return file
        raise RuntimeError("No last.cpkt found!")

    # Remove .suffix and the -
    cktp_metrics = [str(file.stem).split("-") for file in cktp_files if not file.stem in ["last", "final", "best"]]

    # Extract metrics and store them in dicts
    metric_dicts = []
    for metric_list in cktp_metrics:
        metric_dict = {}
        for entry in metric_list:
            metric_name, value = entry.split("=")
            metric_dict[metric_name] = float(value)
        metric_dicts.append(metric_dict)

    # Only keep searched metric
    relevant_metrics = [metric_dict[by_metric] for metric_dict in metric_dicts]

    # Search for min or max of metric
    extremum: float = max(relevant_metrics) if mode_max else min(relevant_metrics)

    # Find file
    idx = relevant_metrics.index(extremum)

    return cktp_files[idx]

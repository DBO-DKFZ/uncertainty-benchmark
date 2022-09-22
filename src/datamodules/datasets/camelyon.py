# STL
import os
import random
import typing
from operator import itemgetter
import json
from pathlib import Path
from typing import Union, Callable, Optional, Sequence
import multiprocessing as mp
from tqdm import tqdm
import colorsys

# Common
import PIL.Image

# import cv2
# cv2.setNumThreads(1)  # limit num_threads to 1 thread per worker
import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import DBSCAN
import sklearn.metrics as skmetrics

# Torch
import torch
import torchvision
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    CenterCrop,
    Resize,
    Grayscale,
)
from torch.utils.data import Dataset, Sampler, DataLoader

# Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from ...metrics.metrics import compute_confidence, normed_entropy

DEBUG = False

_MULTIPROCESS = True
"""Set to True to enable multiprocessing in this file. Used for JSON parsing, as well as slide visualization."""


def _createSlideVisualization(
    self,
    slide_name: str,
    pred_list: torch.Tensor,
    tile_size: int,
    tile_pix_size: int,
    overlap: float,
    processing_level: int,
    hue_values: Sequence,
    remove_non_tissue: bool = False,
    background_clip: Optional[float] = None,
):

    overlap_pix = int(tile_size * overlap)

    # TODO: Fix annotation overlap

    # So far we just round down to the nearest pixel location.  So if tile_size//overlap_pix == tile_pix_size everything works fine.
    if overlap_pix != 0 and (tile_size % overlap_pix != 0 or tile_pix_size % (tile_size // overlap_pix) != 0):
        print(
            "Specified overlap together with tile_pix_size will lead to artifacts. Set tile_pix_size = tile_size//overlap_pix, in case this results in an integer, to avoid this problem."
        )

    # Get relevant tiles for slide
    df = self.data
    indices = df.index[df["slide_name"] == slide_name].tolist()
    relevant_tiles = df["patch_path"][indices]
    relevant_labels = df["label"][indices]
    relevant_labels = relevant_labels.replace(self.LABEL_DICT).tolist()
    tile_x = df["x_pos"][indices].to_numpy()
    tile_y = df["y_pos"][indices].to_numpy()
    tile_locs = np.hstack((tile_x[:, None], tile_y[:, None]))
    relevant_predictions = pred_list[indices]

    tumor_conf = relevant_predictions[:, 1]
    no_tumor_conf = relevant_predictions[:, 0]
    entropy = normed_entropy(relevant_predictions, num_classes=self.num_classes)
    conf, pred = torch.max(torch.tensor(relevant_predictions), dim=1)
    acc = pred == torch.tensor(relevant_labels)

    # Extract bounding box
    min_x = np.min(tile_locs[:, 0])
    max_x = np.max(tile_locs[:, 0])
    min_y = np.min(tile_locs[:, 1])
    max_y = np.max(tile_locs[:, 1])

    # Need the +tile_size, because the coordinates are tracked at the top left. So we have one more tile.
    img_dims = [
        ((max_x + tile_size - min_x) // tile_size) * tile_pix_size,
        ((max_y + tile_size - min_y) // tile_size) * tile_pix_size,
    ]

    # Build image
    class_img = torch.zeros(img_dims)
    pred_img = torch.zeros(img_dims)
    acc_img = torch.zeros(img_dims)
    conf_img = torch.zeros(img_dims)

    coverage_mask = torch.zeros(img_dims)

    # Get background image
    org_img_transforms = Compose([ToTensor()])
    thumbnail_path = self.path / slide_name / ("thumbnail." + self.file_format)
    assert thumbnail_path.exists()
    thumbnail = org_img_transforms(PIL.Image.open(thumbnail_path)).transpose(1, 2)

    # Cutout relevant part
    scaling_factor = 2**processing_level
    thumbnail = Resize(img_dims)(
        thumbnail[
            :,
            min_x // scaling_factor : (max_x + tile_size) // scaling_factor,
            min_y // scaling_factor : (max_y + tile_size) // scaling_factor,
        ].unsqueeze(0)
    ).squeeze(0)
    thumbnail = thumbnail.permute([1, 2, 0])
    thumbnail_hsv = torch.tensor(matplotlib.colors.rgb_to_hsv(thumbnail))
    background_img = thumbnail_hsv[:, :, 2]

    def __splat_image(image, loc, value, add=False):
        if not add:
            image[loc[0] : loc[0] + tile_pix_size, loc[1] : loc[1] + tile_pix_size] = value
        else:
            image[loc[0] : loc[0] + tile_pix_size, loc[1] : loc[1] + tile_pix_size] += value

    for tile_idx in range(len(relevant_tiles)):

        loc = tile_locs[tile_idx]
        loc_x = ((((loc[0]) - min_x) * tile_pix_size)) // tile_size
        loc_y = ((((loc[1]) - min_y) * tile_pix_size)) // tile_size

        __splat_image(class_img, [loc_x, loc_y], relevant_labels[tile_idx])
        __splat_image(pred_img, [loc_x, loc_y], pred[tile_idx])
        __splat_image(acc_img, [loc_x, loc_y], acc[tile_idx], True)
        __splat_image(conf_img, [loc_x, loc_y], tumor_conf[tile_idx], True)

        __splat_image(coverage_mask, [loc_x, loc_y], 1, True)

    binary_mask = coverage_mask > 0

    # Avoid completely dark background
    if background_clip is not None:
        background_img = torch.clamp_min_(background_img, background_clip)

    # Average overlapping tiles
    # Dont divide by 0.

    class_img[binary_mask] /= coverage_mask[binary_mask]
    acc_img[binary_mask] /= coverage_mask[binary_mask]
    pred_img[binary_mask] /= coverage_mask[binary_mask]
    conf_img[binary_mask] /= coverage_mask[binary_mask]

    if remove_non_tissue:
        background_img[~binary_mask] = 0

    def __build_image(hue_channel):
        hsv_image = torch.zeros(img_dims + [3])

        # V-channel
        hsv_image[:, :, 2] = background_img

        # S-channel
        s_channel = hsv_image[:, :, 1]
        s_channel[binary_mask] = 1

        # H-channel
        hsv_image[:, :, 0] = hue_channel

        rgb_image = matplotlib.colors.hsv_to_rgb(hsv_image)

        return rgb_image

    def __build_class_image(class_image, hue_values):
        h_channel = torch.zeros_like(class_image)
        h_channel[class_image < 0.5] = hue_values[0]
        h_channel[class_image >= 0.5] = hue_values[1]
        return __build_image(h_channel)

    def __build_acc_image(acc_image, hue_values):
        h_channel = hue_values[0] + acc_image * hue_values[1]
        return __build_image(h_channel)

    def __build_conf_image(conf_image, hue_values, num_classes=2):
        # Scale the range [1/num_classes, 1] to [0,1]
        scaled_conf_image = (
            (conf_image - (1 / num_classes)) * num_classes / (num_classes - 1)
        )  # Equivalent to 1/(1-(1/num_classes))
        h_channel = hue_values[0] + scaled_conf_image * hue_values[1]
        return __build_image(h_channel)

    class_image = __build_class_image(class_img, hue_values)
    acc_image = __build_acc_image(acc_img, hue_values)
    pred_img = __build_class_image(pred_img, hue_values)
    # conf_image = __build_conf_image(conf_img, hue_values)
    conf_image = conf_img.numpy()
    conf_masked = np.ma.masked_where(~binary_mask, conf_image)
    # plt.imshow(background_img, cmap=plt.cm.gray)
    # plt.imshow(conf_masked, cmap=plt.cm.plasma, alpha=0.8)
    # plt.show()

    # slidename = self.slide_folders[slide_idx].parts[-1]

    image_dict = {
        "slidename": slide_name,
        "image": thumbnail,
        "background": background_img,
        "class": class_image,
        "pred": pred_img,
        "acc": acc_image,
        "conf": conf_masked,
    }

    return image_dict


def _generate_slide_predicition(
    self,
    slide_name: str,
    pred_list: torch.Tensor,
    tile_size: int,
    overlap: float,
    tumor_thresh: float = 0.5,
    approach: str = "convolution",
):
    # tile_pix_size is fixed at 1 for slide predictions
    tile_pix_size = 1

    overlap_pix = int(tile_size * overlap)

    # TODO: Fix annotation overlap

    # So far we just round down to the nearest pixel location.  So if tile_size//overlap_pix == tile_pix_size everything works fine.
    if overlap_pix != 0 and (tile_size % overlap_pix != 0 or tile_pix_size % (tile_size // overlap_pix) != 0):
        print(
            "Specified overlap together with tile_pix_size will lead to artifacts. Set tile_pix_size = tile_size//overlap_pix, in case this results in an integer, to avoid this problem."
        )

    # Get relevant tiles for slide
    df = self.data
    indices = df.index[df["slide_name"] == slide_name].tolist()
    relevant_tiles = df["patch_path"][indices]
    relevant_labels = df["label"][indices]
    relevant_labels = relevant_labels.replace(self.LABEL_DICT).tolist()
    tile_x = df["x_pos"][indices].to_numpy()
    tile_y = df["y_pos"][indices].to_numpy()
    tile_locs = np.hstack((tile_x[:, None], tile_y[:, None]))
    relevant_predictions = pred_list[indices]
    tumor_pred = relevant_predictions[:, 1]

    # Extract bounding box
    min_x = np.min(tile_locs[:, 0])
    max_x = np.max(tile_locs[:, 0])
    min_y = np.min(tile_locs[:, 1])
    max_y = np.max(tile_locs[:, 1])

    # Need the +tile_size, because the coordinates are tracked at the top left. So we have one more tile.
    img_dims = [
        ((max_x + tile_size - min_x) // tile_size) * tile_pix_size,
        ((max_y + tile_size - min_y) // tile_size) * tile_pix_size,
    ]

    # Build image
    pred_img = torch.zeros(img_dims)

    def __splat_image(image, loc, value, add=False):
        if not add:
            image[loc[0] : loc[0] + tile_pix_size, loc[1] : loc[1] + tile_pix_size] = value
        else:
            image[loc[0] : loc[0] + tile_pix_size, loc[1] : loc[1] + tile_pix_size] += value

    for tile_idx in tqdm(range(len(relevant_tiles))):

        loc = tile_locs[tile_idx]
        loc_x = ((((loc[0]) - min_x) * tile_pix_size)) // tile_size
        loc_y = ((((loc[1]) - min_y) * tile_pix_size)) // tile_size

        __splat_image(pred_img, [loc_x, loc_y], 1 if tumor_pred[tile_idx] > tumor_thresh else 0)

    slide_pred = 0
    if approach == "convolution":
        # if torch.cuda.is_available():
        #     device = "cuda:0"
        # else:
        device = "cpu"  # Compute on GPU did not show speed increase
        # Apply 5x5 convolution with weight 1 to sum tumor probabilities
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, device=device)
        conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight))
        input = pred_img[None, None, :, :]  # Convert image to (N, C, H, W) format
        with torch.no_grad():
            output = conv(input.to(device))

        output = np.squeeze(output.cpu().numpy())
        max_tumor = np.max(output)

        if DEBUG:
            print(max_tumor)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.imshow(pred_img)
            ax2.imshow(output)
            plt.show()

        if max_tumor > 20:
            slide_pred = 1

    elif approach == "dbscan":
        row, col = np.where(pred_img == 1)
        # rows are y, colums are x coordinates
        coords = np.hstack((col[:, None], row[:, None]))
        db = DBSCAN(eps=3, min_samples=10).fit(coords)
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        if DEBUG:
            print("Estimated number of clusters: %d" % n_clusters)
            print("Estimated number of noise points: %d" % n_noise)
            plt.imshow(pred_img)
            plt.show()

        if n_clusters > 0:
            slide_pred = 1

    else:
        raise RuntimeError("Approach must be either convolution or dbscan")

    return slide_pred


class Camelyon(Dataset):
    LABEL_DICT = {"non_tumor": 0, "tumor": 1, "unlabeled": np.NAN}
    MEAN_STD = [
        torch.Tensor([0.6054, 0.4499, 0.6005]),
        torch.Tensor([0.2284, 0.2463, 0.1928]),
    ]

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        split: str = "train",
        transformlib: str = "torchvision",
        transforms=None,
        target_transforms: Callable = None,
        split_setup=(0.6, 0.2, 0.2),
    ):
        super().__init__()

        assert split in ["train", "val", "test", "all"]
        assert sum(split_setup) == 1

        if not isinstance(path, Path):
            path = Path(path)

        self.path = path
        self.name = name
        self.split = split
        # self.centers = [0, 1, 2, 3, 4] if centers is None else centers
        assert transformlib in ["torchvision", "albumentations"]
        self.transformlib = transformlib
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.split_setup = split_setup

        config_p = path / "config.json"

        if not config_p.exists():
            raise RuntimeError("Did not find config.json in path directory.")

        config = json.load(open(config_p.absolute(), "r"))

        # We have 2 classes for the tiles (pos, neg)
        self.num_classes = len(config["label_dict"])

        self.overlap = config["overlap"]
        self.tile_size = config["patch_size"]
        self.file_format = config["output_format"]
        self.process_level = config["processing_level"]

        slide_inf_p = path / "slide_information.csv"

        if not slide_inf_p.exists():
            raise RuntimeError("Did not find slide_information.csv in path directory.")

        slide_inf = pd.read_csv(slide_inf_p)
        self.id_test_slidenames = []
        slide_indices = self._infer_slide_indices(slide_inf)

        slide_folders = [fol.absolute() for fol in path.iterdir() if fol.is_dir()]
        # Sort by slide name
        slide_folders = sorted(slide_folders, key=lambda x: x.parts[-1])

        selected_slides = []
        for idx in slide_indices:
            selected_slides.append(slide_folders[idx])

        # Load tile_information from selected_slides
        dfs = []
        # print("Loading data from csv's")
        for fol in selected_slides:
            csv_p = fol / ("tile_information.csv")
            assert csv_p.exists()
            dfs.append(pd.read_csv(csv_p))
        df = pd.concat(dfs)
        df = df.reset_index(drop=True)

        # If necessary, split dataset by tiles
        if split == "train":
            self.data = self._split_dataset(df, "train")
        elif split == "val":
            self.data = self._split_dataset(df, "val")
        else:
            self.data = df
        # self.class_counts = class_counts

    def _infer_slide_indices(self, slide_inf: pd.DataFrame) -> list:
        # Extract slides from selected centers from slide_inf
        splits = [int(x * len(slide_inf)) for x in self.split_setup]
        splits[2] += len(slide_inf) - sum(splits)
        slide_indices = list(range(len(slide_inf)))
        class_balance = slide_inf["frac"][slide_indices].to_numpy()

        # Extract test slides by using slide statistics
        if self.split != "all":
            mean_balance = np.mean(class_balance)
            diff = np.abs(class_balance - mean_balance)
            test_slide_indices = np.argsort(diff)[: splits[2]]
            test_slide_indices = sorted(list(test_slide_indices))
            self.id_test_slidenames = slide_inf["slide_name"][test_slide_indices].to_list()
            if self.split in ["train", "val"]:
                # Remove selected test slides
                indices = sorted(test_slide_indices, reverse=True)  # Important, otherwise running out of bounds
                for idx in indices:
                    slide_indices.pop(idx)
            elif self.split == "test":
                slide_indices = test_slide_indices

        return slide_indices

    def _split_dataset(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        train_val_split = torch.Tensor((self.split_setup[0], self.split_setup[1]))
        train_val_split = train_val_split / train_val_split.sum()
        df_tumor = df[df["label"] == "tumor"]
        df_non_tumor = df[df["label"] == "non_tumor"]
        # Shuffle tiles
        df_tumor = df_tumor.sample(frac=1, random_state=np.random.RandomState(42))
        df_non_tumor = df_non_tumor.sample(frac=1, random_state=np.random.RandomState(42))
        len_tumor_train = int(len(df_tumor) * train_val_split[0])
        len_non_tumor_train = int(len(df_non_tumor) * train_val_split[0])
        if split == "train":
            df_tumor = df_tumor[:len_tumor_train]
            df_non_tumor = df_non_tumor[:len_non_tumor_train]
        elif split == "val":
            df_tumor = df_tumor[len_tumor_train:]
            df_non_tumor = df_non_tumor[len_non_tumor_train:]
        else:
            raise RuntimeError("Dataset split only possible for train and val split")
        df = pd.concat([df_non_tumor, df_tumor])
        df = df.reset_index(drop=True)
        return df

    def _threshold_tumor_tiles(self, df: pd.DataFrame, tumor_threshold: float) -> pd.DataFrame:
        if "tumor_coverage" in df.columns:
            df_tumor = df[df["label"] == "tumor"]
            df_non_tumor = df[df["label"] == "non_tumor"]
            df_tumor = df_tumor[df_tumor["tumor_coverage"] >= tumor_threshold]
            df = pd.concat([df_non_tumor, df_tumor])
            df = df.reset_index(drop=True)
        else:
            raise RuntimeError("Data does not contain information on tumor coverage per tile")
        return df

    def get_id_test_slidenames(self):
        return self.id_test_slidenames

    def generate_subset(self, frac: float, random_state: Union[int, np.random.RandomState]):
        df = self.data
        df = df.sample(frac=frac, random_state=random_state)
        df = df.reset_index(drop=True)
        self.data = df
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> PIL.Image.Image:

        # Combine data elements to absolute path
        slide_name = self.data["slide_name"][item]
        patch_path = self.data["patch_path"][item]
        img_path = self.path / slide_name / patch_path
        img = PIL.Image.open(img_path)

        label = self.data["label"][item]
        # Map label string to int
        label = self.LABEL_DICT[label]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            label = self.target_transforms(label)

        return img, label

    # Optional: Used for over- and under-sampling
    def getlabel(self, item: int) -> int:
        label = self.data["label"][item]
        # Map label string to int
        label = self.LABEL_DICT[label]
        return label

    # Optional: Returns original image without transformations
    def getImage(self, item: int) -> PIL.Image.Image:

        slide_name = self.data["slide_name"][item]
        patch_path = self.data["patch_path"][item]
        img_path = self.path / slide_name / patch_path
        img = PIL.Image.open(img_path)
        return img

    @staticmethod
    def parallel_json(idx: int, json_f: Union[str, Path]) -> tuple:

        f = json.load(open(str(json_f), "r"))

        # Process each tile
        tiles_classes, tmp_data = Camelyon17._parseTiles(f, json_f)

        return idx, tiles_classes, tmp_data

    @staticmethod
    def _parseTiles(
        json_c: dict, json_p: Path, labels: tuple[str] = ("non_tumor", "tumor")
    ) -> tuple[torch.Tensor, list[list[Path, int, dict]]]:

        num_classes = len(labels)
        mapping_dict = {name: idx for idx, name in enumerate(labels)}

        class_counts = torch.zeros(num_classes, dtype=torch.int)
        data = []

        num_tiles = len(json_c)

        for idx in range(num_tiles):

            dict_entry = json_c[str(idx)]

            x = dict_entry["x_pos"]
            y = dict_entry["y_pos"]

            size = dict_entry["patch_size"]
            c = mapping_dict[dict_entry["label"]]
            tile_name = Path(dict_entry["patch_path"])

            folder = json_p.parents[0].absolute()

            tile_path = folder / tile_name

            assert tile_path.exists()

            # Get rid of useless information
            data_dict = {"x": x, "y": y}

            # Increment the class_counts only after swapping.
            class_counts[c] += 1

            d_entry = [tile_path, c, data_dict]

            data.append(d_entry)

        return class_counts, data

    def build_slide_lvl_images(
        self,
        pred_list: Union[torch.Tensor, list[torch.Tensor]],
        tile_render_size: Optional[Union[int, typing.Iterable[int]]] = 2,
        select_slides: Optional[list[int]] = None,
    ):

        # if not self.split_by_slides and self.split != "all":
        #     raise NotImplementedError(
        #         f"Called build_slide_lvl_images, while split_by_slide is set to {self.split_by_slides} and split not set to all."
        #     )
        # slide_idxs = []
        # if select_slides is not None:
        #     for idx in select_slides:
        #         assert 0 <= idx < len(self.slide_folders)
        #     slide_idxs = select_slides
        # else:
        #     slide_idxs = list(range(len(self.slide_folders)))

        assert len(pred_list) == len(self)

        df = self.data
        selected_slides = df["slide_name"].unique().tolist()

        if select_slides is not None:
            tmp = []
            for slide_idx in select_slides:
                tmp.append(selected_slides[slide_idx])

            selected_slides = tmp

        # In HSV space
        red_color = colorsys.rgb_to_hsv(1.0, 0.0, 0.0)
        blue_color = colorsys.rgb_to_hsv(0.0, 0.0, 1.0)
        green_color = colorsys.rgb_to_hsv(0.0, 1.0, 0.0)

        # Use hue values for "magenta" and "cyan"
        hue_values = [300 / 360, 180 / 360]

        # Compute start idx of every slide in data array
        # cum_idx = torch.tensor(self.slide_info["index"])

        proc_pool = mp.Pool(processes=mp.cpu_count())

        args = [
            [
                self,
                selected_slides[idx],
                pred_list,
                self.tile_size,
                tile_render_size,
                self.overlap,
                self.process_level,
                hue_values,
            ]
            for idx in range(len(selected_slides))
        ]

        if _MULTIPROCESS:
            images = proc_pool.starmap(_createSlideVisualization, args)

            proc_pool.close()
            proc_pool.join()
        else:
            images = []

            for idx in range(len(selected_slides)):
                images.append(_createSlideVisualization(*args[idx]))

        return images

    def generate_slide_prediction(
        self,
        pred_list: Union[torch.Tensor, list[torch.Tensor]],
        select_slides: Optional[list[int]] = None,
        tumor_thresh: float = 0.5,
        approach: str = "convolution",
    ):
        assert len(pred_list) == len(self)

        df = self.data
        selected_slides = df["slide_name"].unique().tolist()

        if select_slides is not None:
            tmp = []
            for slide_idx in select_slides:
                tmp.append(selected_slides[slide_idx])

            selected_slides = tmp

        args = [
            [
                self,
                selected_slides[idx],
                pred_list,
                self.tile_size,
                self.overlap,
                tumor_thresh,
                approach,
            ]
            for idx in range(len(selected_slides))
        ]

        slide_preds = []

        for idx in range(len(selected_slides)):
            slide_preds.append(_generate_slide_predicition(*args[idx]))

        return slide_preds


class Camelyon16(Camelyon):
    LABEL_DICT = {"non_tumor": 0, "tumor": 1}
    MEAN_STD = [
        torch.Tensor([0.6054, 0.4499, 0.6005]),
        torch.Tensor([0.2284, 0.2463, 0.1928]),
    ]

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        split: str = "train",
        transformlib: str = "torchvision",
        transforms=None,
        target_transforms: Callable = None,
        split_setup=(0.6, 0.2, 0.2),
        tumor_threshold: float = 0.25,
    ):
        super().__init__(
            path=path,
            name=name,
            split=split,
            transformlib=transformlib,
            transforms=transforms,
            target_transforms=target_transforms,
            split_setup=split_setup,
        )
        self.data = self._threshold_tumor_tiles(self.data, tumor_threshold)


class Camelyon17(Camelyon):

    LABEL_DICT = {"non_tumor": 0, "tumor": 1}
    MEAN_STD = [
        torch.Tensor([0.6054, 0.4499, 0.6005]),
        torch.Tensor([0.2284, 0.2463, 0.1928]),
    ]

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        split: str = "train",
        centers: Union[list[int], tuple[int]] = None,
        transformlib: str = "torchvision",
        transforms=None,
        target_transforms: Callable = None,
        split_setup=(0.6, 0.2, 0.2),
        tumor_threshold: float = 0.25,
    ):
        self.centers = centers
        self.slides_per_center = 10  # Number of slides per center is always 10
        super().__init__(
            path=path,
            name=name,
            split=split,
            transformlib=transformlib,
            transforms=transforms,
            target_transforms=target_transforms,
            split_setup=split_setup,
        )
        self.data = self._threshold_tumor_tiles(self.data, tumor_threshold)

    def _infer_slide_indices(self, slide_inf: pd.DataFrame) -> list:
        # Extract slides from selected centers from slide_inf
        spc = self.slides_per_center
        splits = [int(x * self.slides_per_center) for x in self.split_setup]
        slide_indices = []
        for c in self.centers:
            slide_indices += list(range(c * spc, c * spc + spc))
        class_balance = slide_inf["frac"][slide_indices]

        # Extract test slides by using slide statistics
        if self.split != "all":
            indices = []
            for i, c in enumerate(self.centers):
                center_indices = list(range(c * spc, c * spc + spc))
                center_class_balance = class_balance[center_indices].to_numpy()
                mean_balance = np.mean(center_class_balance)
                diff = np.abs(center_class_balance - mean_balance)
                idx = np.argsort(diff)[: splits[2]]
                idx += i * spc  # Increase index dependent on center
                indices += list(idx)
            indices = sorted(indices)
            test_slide_indices = []
            for idx in indices:
                test_slide_indices.append(slide_indices[idx])
            self.id_test_slidenames = slide_inf["slide_name"][test_slide_indices].to_list()
            if self.split in ["train", "val"]:
                # Remove selected test slides
                indices = sorted(indices, reverse=True)  # Important, otherwise running out of bounds
                for idx in indices:
                    slide_indices.pop(idx)
            elif self.split == "test":
                slide_indices = test_slide_indices

        return slide_indices


class Camelyon17LabelNoise(Camelyon17):

    LABEL_DICT = {"non_tumor": 0, "tumor": 1}
    MEAN_STD = [
        torch.Tensor([0.6054, 0.4499, 0.6005]),
        torch.Tensor([0.2284, 0.2463, 0.1928]),
    ]

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        split: str = "train",
        centers: Union[list[int], tuple[int]] = None,
        transformlib: str = "torchvision",
        transforms=None,
        target_transforms: Callable = None,
        split_setup=(0.6, 0.2, 0.2),
        uniform_label_flip=None,
        edge_label_flip=None,
        tumor_threshold=0.0,
        random_seed=8350204,
    ):

        assert edge_label_flip in [None, "proportional"] or isinstance(edge_label_flip, float)
        assert not (
            (edge_label_flip is not None and edge_label_flip > 0)
            and (uniform_label_flip is not None and uniform_label_flip > 0)
        ), f"You cannot set edge_label_flip ({edge_label_flip}) and uniform_label_flip ({uniform_label_flip}) to a positive value."

        assert uniform_label_flip is None or 0.0 <= uniform_label_flip <= 0.5

        self.edge_label_flip = edge_label_flip
        self.uniform_label_flip = uniform_label_flip

        super().__init__(
            path=path,
            name=name,
            split=split,
            centers=centers,
            transformlib=transformlib,
            transforms=transforms,
            target_transforms=target_transforms,
            split_setup=split_setup,
            tumor_threshold=0.0,
        )

        torch_generator = torch.Generator().manual_seed(random_seed)
        self.random_uniform_mask = torch.rand(len(self), generator=torch_generator) > 1.0 - (
            0.0 if self.uniform_label_flip is None else self.uniform_label_flip
        )
        self.random_edge_mask = torch.rand(len(self), generator=torch_generator) > 1.0 - (
            0.0 if self.edge_label_flip is None else self.edge_label_flip
        )

    def __getitem__(self, item):

        img, label = super().__getitem__(item)

        if self.edge_label_flip is not None and label == 1 and self.random_edge_mask[item]:
            label = int(not label)

            # if self.edge_label_flip == "proportional" and edge_label_realization > self.data["tumor_coverage"][item]:
            #    label = int(not label)
            # if self.random_edge_mask[item]:
            #    label = int(not label)

        if self.random_uniform_mask[item]:
            label = int(not label)

        return img, label


class Camelyon17WholeSlide(Camelyon):
    MEAN_STD = [
        torch.Tensor([0.6054, 0.4499, 0.6005]),
        torch.Tensor([0.2284, 0.2463, 0.1928]),
    ]

    def __init__(
        self,
        path: Union[str, Path],
        name: str,
        centers: Union[list[int], tuple[int]] = None,
        transformlib: str = "torchvision",
        transforms=None,
        target_transforms: Callable = None,
    ):
        self.centers = centers
        self.slides_per_center = 90
        super().__init__(
            path=path,
            name=name,
            split="all",
            transformlib=transformlib,
            transforms=transforms,
            target_transforms=target_transforms,
            split_setup=[0, 0, 1],
        )

        stage_labels = pd.read_csv(Path(path) / "stage_labels.csv")

        # Clean up stage_labels

        keep = stage_labels["patient"].apply(lambda x: ".zip" not in x)
        stage_labels = stage_labels.loc[keep]
        stage_labels["patient"] = stage_labels["patient"].map(lambda x: x.split(".")[0])
        stage_labels = stage_labels.rename(columns={"patient": "slide_name"})

        # Drop the 50 removed slides
        self.data = pd.merge(self.data, stage_labels, how="left", left_on="slide_name", right_on="slide_name")

    def _infer_slide_indices(self, slide_inf: pd.DataFrame) -> list:
        # Extract slides from selected centers from slide_inf
        spc = self.slides_per_center
        slide_indices = []
        for c in self.centers:
            slide_indices += list(range(c * spc, c * spc + spc))

        return slide_indices

    def get_slide_label(self, index_name: Union[str, int]):
        if isinstance(index_name, str):
            label = None
        elif isinstance(index_name, int):
            label = None
        else:
            raise RuntimeError()
        return label


class BalanceDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, sampling_factor: Union[str, float]):
        super().__init__()

        self.dataset = dataset
        if hasattr(dataset, "name"):
            self.name = dataset.name
        else:
            self.name = None

        if sampling_factor is None:
            raise Exception("sampling_factor is None")

        if sampling_factor == "oversampling":
            sampling_factor = 1.0
        elif sampling_factor == "undersampling":
            sampling_factor = 0.0

        org_size = len(self.dataset)

        if hasattr(self.dataset, "getlabel"):
            labels = [self.dataset.getlabel(i) for i in range(org_size)]
        else:
            labels = [self.dataset[i][1] for i in range(org_size)]

        self.num_classes = len(set(labels))
        self.labels = [[] for _ in range(self.num_classes)]

        for i, label in enumerate(labels):
            self.labels[label].append(i)

        num_labels = [len(i) for i in self.labels]
        min_size = min(num_labels)
        max_size = max(num_labels)

        # Linear interpolate the class size between the largest and smallest class
        if 0.0 <= sampling_factor <= 1.0:
            inter_class_distance = abs(max_size - min_size)
            self.class_size = min_size + int(sampling_factor * inter_class_distance)
        # Downsample the largest class by a factor
        elif -1.0 < sampling_factor < 0.0:
            self.class_size = int(max_size * -sampling_factor)
        # Upsample the smallest class by a factor
        elif sampling_factor < -1.0:
            self.class_size = int(min_size * -sampling_factor)
        else:
            raise NotImplementedError(
                "Called sampling factor with value -1.0 (behaviour undefined) or a float greater than 1.0"
            )

        self.selected_indices = self.__build_indices()

    def __build_indices(self):

        indices = []

        for c in range(self.num_classes):
            # Determine if we need to over- or undersample
            class_l = self.labels[c]
            num_in_class = len(class_l)

            # We have fewer actual labels than required. We need to oversample
            if num_in_class <= self.class_size:
                reps = self.class_size // num_in_class
                sample = self.class_size % num_in_class
                assert sample == self.class_size - reps * num_in_class
            # We need to undersample
            else:
                reps = 0
                sample = self.class_size

            new_ind = (class_l * reps) + sorted(random.sample(class_l, sample))
            indices += new_ind

        assert len(indices) == self.class_size * self.num_classes

        return indices

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, index):
        return self.dataset[self.selected_indices[index]]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    dataset = Camelyon17(
        path=args.data_dir,
        name="train",
        split="train",
        centers=[0, 1, 2, 3],
        transforms=ToTensor(),
    )

    balanced_dataset = BalanceDatasetWrapper(
        dataset=dataset,
        sampling_factor=0.0,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
    )

    # Code from https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c
    print("Computing mean and std...")
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(dataloader):
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    print("Mean: " + str(mean))
    print("Std: " + str(std))

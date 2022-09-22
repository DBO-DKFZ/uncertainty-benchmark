# STL
import os
import random
import typing
from operator import itemgetter
import json
from pathlib import Path
from typing import Union, Callable, Optional, Sequence
import multiprocessing as mp
import colorsys

# Common
import PIL.Image
import matplotlib.colors
from matplotlib.image import thumbnail
from tqdm import tqdm
from PIL import Image
import numpy as np

# Torch
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, Sampler, DataLoader

# Lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

# Own
from .datasets.camelyon import (
    Camelyon16,
    Camelyon17,
    Camelyon17WholeSlide,
    Camelyon17LabelNoise,
    BalanceDatasetWrapper,
)

import warnings
from pytorch_lightning.utilities.rank_zero import LightningDeprecationWarning

# train_transform and test_transform raise Deprecation Warnings in lightning... -.- Stupid getters and setters
warnings.simplefilter(
    "ignore",
    category=LightningDeprecationWarning,
    lineno=0,
    append=False,
)
warnings.simplefilter("ignore", category=UserWarning, lineno=0, append=False)


class DiscreteRotationTransform:
    """Rotate by one of the given angles."""

    # c.f. https://pytorch.org/vision/stable/transforms.html

    def __init__(self, angles: Union[Sequence[int], Sequence[float]]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return torchvision.transforms.functional.rotate(x, angle)


# Set up torchvision transforms according to Tellez et al., 2019
tv_base_transforms = [T.ToTensor(), T.Normalize(*Camelyon17.MEAN_STD)]

tv_basic_transforms = [
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    DiscreteRotationTransform(angles=[-90, 0, 90]),
]

tv_morphology_transforms = [
    T.RandomAffine(degrees=0, scale=(1.0, 1.2)),
    T.GaussianBlur(kernel_size=(5, 5), sigma=(1e-9, 0.1)),
]

tv_hsv_light_transfroms = [
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.1, hue=0.1),
]

tv_hsv_strong_transforms = [
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=1.0, hue=1.0),
]

tv_train = T.Compose(
    [
        T.RandomCrop(size=224),
        *tv_basic_transforms,
        *tv_morphology_transforms,
        *tv_hsv_light_transfroms,
        *tv_base_transforms,
    ]
)
tv_val = T.Compose([T.CenterCrop(size=224)] + tv_base_transforms)


class CamelyonDataModule(LightningDataModule):
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        split_setup: list,
        sampling_factor: Union[str, float],
        transformlib: str,
        train_transforms: torchvision.transforms,
        val_transforms: torchvision.transforms,
        test_transforms: torchvision.transforms,
        shuffle_train: bool,
        balance_train: bool,
        val_subset: float,
        balance_val: bool,
        balance_test: bool,
    ):
        super().__init__()

        self.path = Path(path)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.split_setup = split_setup
        self.sampling_factor = sampling_factor

        self.transformlib = transformlib
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.shuffle = shuffle_train
        self.balance_train = balance_train
        self.balance_val = balance_val
        self.balance_test = balance_test

        if 0.0 < val_subset <= 1.0:
            self.val_subset = val_subset
        else:
            raise RuntimeError("Invalid value for val_subset")

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        pass  # Needs to be defined in subclasses

    # return the dataloader for each split
    def train_dataloader(self):

        if self.balance_train:
            camelyon_train = BalanceDatasetWrapper(self.camelyon_train, self.sampling_factor)
        else:
            camelyon_train = self.camelyon_train

        camelyon_train = DataLoader(
            camelyon_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return camelyon_train

    def val_dataloader(self):

        if self.balance_val:
            camelyon_val = BalanceDatasetWrapper(self.camelyon_val, self.sampling_factor)
        else:
            camelyon_val = self.camelyon_val

        camelyon_val = DataLoader(
            camelyon_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return camelyon_val

    def test_dataloader(self):

        if self.balance_test:
            camelyon_test = BalanceDatasetWrapper(self.camelyon_test, self.sampling_factor)
            camelyon_ood = BalanceDatasetWrapper(self.camelyon_ood, self.sampling_factor)
        else:
            camelyon_test = self.camelyon_test
            camelyon_ood = self.camelyon_ood

        camelyon_test = DataLoader(
            camelyon_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        camelyon_ood = DataLoader(
            camelyon_ood,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return camelyon_test, camelyon_ood


class Camelyon16DataModule(CamelyonDataModule):
    def __init__(
        self,
        path: Union[str, Path],
        path_Cam17: Union[str, Path],
        batch_size: int,
        num_workers: int,
        tumor_threshold: float,
        split_setup: list,
        sampling_factor: Union[str, float],
        transformlib: str,
        train_transforms: torchvision.transforms,
        val_transforms: torchvision.transforms,
        test_transforms: torchvision.transforms,
        shuffle_train: bool,
        balance_train: bool,
        val_subset: float,
        balance_val: bool,
        balance_test: bool,
    ):
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            split_setup=split_setup,
            sampling_factor=sampling_factor,
            transformlib=transformlib,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            shuffle_train=shuffle_train,
            balance_train=balance_train,
            val_subset=val_subset,
            balance_val=balance_val,
            balance_test=balance_test,
        )
        self.tumor_threshold = tumor_threshold
        self.path_Cam17 = path_Cam17

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        if stage in (None, "fit"):
            print("Generating Camelyon16 train dataset")
            self.camelyon_train = Camelyon16(
                self.path,
                name="train",
                split="train",
                transformlib=self.transformlib,
                transforms=self.train_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
            )
            # print("Test slide names: ", self.camelyon_train.get_id_test_slidenames())

        if stage in (None, "fit", "validate", "predict"):
            print("Generating Camelyon16 validation dataset")
            self.camelyon_val = Camelyon16(
                self.path,
                name="val",
                split="val",
                transformlib=self.transformlib,
                transforms=self.val_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
            )
            # print("Test slide names: ", self.camelyon_val.get_id_test_slidenames())
            if 0.0 < self.val_subset < 1.0:
                # Fix random_state to always use the same subset between experiments
                self.camelyon_val = self.camelyon_val.generate_subset(frac=self.val_subset, random_state=42)

        if stage in ["test", "predict"]:
            print("Generating Camelyon16 test dataset")
            self.camelyon_test = Camelyon16(
                self.path,
                name="test",
                split="test",
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )
            print("Generating Camelyon17 OOD dataset")
            self.camelyon_ood = Camelyon17(
                self.path_Cam17,
                name="test_ood",
                split="all",
                centers=[0, 1, 4],
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )


class Camelyon17DataModule(CamelyonDataModule):
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        tumor_threshold: float,
        id_centers: list,
        ood_centers: list,
        split_setup: list,
        sampling_factor: Union[str, float],
        transformlib: str,
        train_transforms: torchvision.transforms,
        val_transforms: torchvision.transforms,
        test_transforms: torchvision.transforms,
        shuffle_train: bool,
        balance_train: bool,
        val_subset: float,
        balance_val: bool,
        balance_test: bool,
    ):
        __centers = [0, 1, 2, 3, 4]
        self.id_centers = id_centers
        if ood_centers is None:
            self.ood_centers = [center for center in __centers if center not in self.id_centers]
        else:
            self.ood_centers = ood_centers

        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            split_setup=split_setup,
            sampling_factor=sampling_factor,
            transformlib=transformlib,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            shuffle_train=shuffle_train,
            balance_train=balance_train,
            val_subset=val_subset,
            balance_val=balance_val,
            balance_test=balance_test,
        )
        self.tumor_threshold = tumor_threshold

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        if stage in (None, "fit"):
            print("Generating Camelyon17 train dataset")
            self.camelyon_train = Camelyon17(
                self.path,
                name="train",
                split="train",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.train_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
            )
            # print("Test slide names: ", self.camelyon_train.get_id_test_slidenames())

        if stage in (None, "fit", "validate", "predict"):
            print("Generating Camelyon17 validation dataset")
            self.camelyon_val = Camelyon17(
                self.path,
                name="val",
                split="val",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.val_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
            )
            # print("Test slide names: ", self.camelyon_val.get_id_test_slidenames())
            if 0.0 < self.val_subset < 1.0:
                # Fix random_state to always use the same subset between experiments
                self.camelyon_val = self.camelyon_val.generate_subset(frac=self.val_subset, random_state=42)

        if stage in ["test", "predict"]:
            print("Generating Camelyon17 test dataset")
            self.camelyon_test = Camelyon17(
                self.path,
                name="test",
                split="test",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )
            print("Generating Camelyon17 OOD dataset")
            self.camelyon_ood = Camelyon17(
                self.path,
                name="test_ood",
                split="all",
                centers=self.ood_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )


@DATAMODULE_REGISTRY
class Camelyon16BaseDataModule(Camelyon16DataModule):
    def __init__(
        self,
        path: Union[str, Path],
        path_Cam17: Union[str, Path],
        batch_size: int,
        num_workers: int,
        tumor_threshold: float = 0.25,
        sampling_factor=None,
        val_subset=1.0,
        transformlib="torchvision",
    ):

        split_setup = (0.7, 0.1, 0.2)
        if transformlib == "torchvision":
            train_transforms = tv_train
            val_transforms = tv_val
            test_transforms = tv_val
        else:
            raise RuntimeError("Provided unknown transformlib")

        super().__init__(
            path,
            path_Cam17,
            batch_size,
            num_workers,
            tumor_threshold,
            split_setup,
            sampling_factor,
            transformlib,
            train_transforms,
            val_transforms,
            test_transforms,
            shuffle_train=True,
            balance_train=True,
            val_subset=val_subset,
            balance_val=False,
            balance_test=False,
        )


@DATAMODULE_REGISTRY
class Camelyon17BaseDataModule(Camelyon17DataModule):
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        tumor_threshold: float = 0.25,
        id_centers: list = [0, 1, 2, 3],
        ood_centers: Union[int, list] = None,
        sampling_factor=None,
        val_subset=1.0,
        transformlib="torchvision",
        augmentations="crop",
    ):
        if type(ood_centers) == int:
            ood_centers = [ood_centers]
        split_setup = (0.6, 0.2, 0.2)
        if transformlib == "torchvision":
            if augmentations == "crop":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        # *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "flip":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "strong":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        *tv_morphology_transforms,
                        *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            else:
                raise RuntimeError("Augmentations must be in [crop, flip, strong]")
            train_transforms = tv_train
            val_transforms = tv_val
            test_transforms = tv_val
        else:
            raise RuntimeError("Provided unknown transformlib")

        super(Camelyon17BaseDataModule, self).__init__(
            path,
            batch_size,
            num_workers,
            tumor_threshold,
            id_centers,
            ood_centers,
            split_setup,
            sampling_factor,
            transformlib,
            train_transforms,
            val_transforms,
            test_transforms,
            shuffle_train=True,
            balance_train=True,
            val_subset=val_subset,
            balance_val=False,
            balance_test=False,
        )


@DATAMODULE_REGISTRY
class Camelyon17WholeSlideDataModule(CamelyonDataModule):
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        id_centers: list,
        transformlib="torchvision",
        augmentations="crop",
        path_Cam17: str = None,
        tumor_threshold: float = 0.0,
        sampling_factor: Union[str, float, int] = None,
        val_subset: float = 1.0,
        ood_centers: list = None,
    ):
        __centers = [0, 1, 2, 3, 4]
        self.id_centers = id_centers
        self.ood_centers = [center for center in __centers if center not in self.id_centers]

        if transformlib == "torchvision":
            if augmentations == "crop":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        # *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "flip":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "strong":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        *tv_morphology_transforms,
                        *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            else:
                raise RuntimeError("Augmentations must be in [crop, flip, strong]")
            train_transforms = tv_train
            val_transforms = tv_val
            test_transforms = tv_val
        else:
            raise RuntimeError("Provided unknown transformlib")

        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            split_setup=[0.0, 0.0, 1.0],
            sampling_factor=None,
            transformlib=transformlib,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            shuffle_train=False,
            balance_train=False,
            val_subset=1.0,
            balance_val=False,
            balance_test=False,
        )

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        if stage in ("fit", "validate"):
            raise RuntimeError("WholeSlideModule does not have a fit and validate split")

        if stage in [None, "test", "predict"]:
            print("Generating Camelyon17 Whole Slide test dataset")
            self.camelyon_test = Camelyon17WholeSlide(
                self.path,
                name="whole_id",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
            )

            print("Generating Camelyon17 Whole Slide OOD dataset")
            self.camelyon_ood = Camelyon17WholeSlide(
                self.path,
                name="whole_ood",
                centers=self.ood_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
            )


@DATAMODULE_REGISTRY
class Camelyon17LabelNoiseDataModule(Camelyon17DataModule):
    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        num_workers: int,
        tumor_threshold: float = 0.00,
        id_centers: list = [0, 1, 2, 3],
        ood_centers: Union[int, list] = None,
        sampling_factor=None,
        val_subset=1.0,
        transformlib="torchvision",
        augmentations="crop",
        edge_label_flip=None,
        uniform_label_flip=0.0,
    ):
        if type(ood_centers) == int:
            ood_centers = [ood_centers]
        split_setup = (0.6, 0.2, 0.2)
        if transformlib == "torchvision":
            if augmentations == "crop":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        # *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "flip":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        # *tv_morphology_transforms,
                        # *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            elif augmentations == "strong":
                tv_train = T.Compose(
                    [
                        T.RandomCrop(size=224),
                        *tv_basic_transforms,
                        *tv_morphology_transforms,
                        *tv_hsv_light_transfroms,
                        *tv_base_transforms,
                    ]
                )
            else:
                raise RuntimeError("Augmentations must be in [crop, flip, strong]")
            train_transforms = tv_train
            val_transforms = tv_val
            test_transforms = tv_val
        else:
            raise RuntimeError("Provided unknown transformlib")

        self.edge_label_flip = edge_label_flip
        self.uniform_label_flip = uniform_label_flip

        super(Camelyon17LabelNoiseDataModule, self).__init__(
            path,
            batch_size,
            num_workers,
            tumor_threshold,
            id_centers,
            ood_centers,
            split_setup,
            sampling_factor,
            transformlib,
            train_transforms,
            val_transforms,
            test_transforms,
            shuffle_train=True,
            balance_train=True,
            val_subset=val_subset,
            balance_val=False,
            balance_test=False,
        )

    def setup(self, stage=None):
        # transforms
        if stage in (None, "fit"):
            print("Generating Camelyon17 train dataset")
            self.camelyon_train = Camelyon17LabelNoise(
                self.path,
                name="train",
                split="train",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.train_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
                edge_label_flip=self.edge_label_flip,
                uniform_label_flip=self.uniform_label_flip,
            )
            # print("Test slide names: ", self.camelyon_train.get_id_test_slidenames())

        if stage in (None, "fit", "validate", "predict"):
            print("Generating Camelyon17 validation dataset")
            self.camelyon_val = Camelyon17LabelNoise(
                self.path,
                name="val",
                split="val",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.val_transforms,
                split_setup=self.split_setup,
                tumor_threshold=self.tumor_threshold,
            )
            # print("Test slide names: ", self.camelyon_val.get_id_test_slidenames())
            if 0.0 < self.val_subset < 1.0:
                # Fix random_state to always use the same subset between experiments
                self.camelyon_val = self.camelyon_val.generate_subset(frac=self.val_subset, random_state=42)

        if stage in ["test", "predict"]:
            print("Generating Camelyon17 test dataset")
            self.camelyon_test = Camelyon17LabelNoise(
                self.path,
                name="test",
                split="test",
                centers=self.id_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )
            print("Generating Camelyon17 OOD dataset")
            self.camelyon_ood = Camelyon17LabelNoise(
                self.path,
                name="test_ood",
                split="all",
                centers=self.ood_centers,
                transformlib=self.transformlib,
                transforms=self.test_transforms,
                split_setup=self.split_setup,
                tumor_threshold=0.0,
            )

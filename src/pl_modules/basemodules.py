import os
from typing import Any, List, Union, Sequence
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn
import torchvision

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, BasePredictionWriter
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

from torchmetrics import Accuracy, ConfusionMatrix, F1Score, AUROC, AveragePrecision, CalibrationError

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..losses import EnsembleLosses
from ..utils.betterargparse import ArgumentParser
import argparse

from ..datamodules.datasets.camelyon import Camelyon17
from torchvision.transforms import Compose, Normalize


DEBUG = False


METRIC_MODE = {
    "val_loss": "min",
    "val_f1": "max",
    "val_b_acc": "max",
}


def transform_batch_outputs(outs):
    if not isinstance(outs[0], list):
        outs = [outs]

    d_idcs = list(range(len(outs)))

    # Store predictions as module attribute
    results = {d_idx: {} for d_idx in d_idcs}

    for d_idx in d_idcs:

        relevant_outs = outs[d_idx]

        rel_keys = set(relevant_outs[0].keys())
        rel_keys.discard("d_idx")

        for key in rel_keys:
            results[d_idx][key] = torch.cat([out[key].cpu() for out in relevant_outs])

    return results


def display_batch(batch: torch.Tensor):
    imgs, labels = batch
    unnorm_transform = Compose(
        [
            Normalize(0, 1 / Camelyon17.MEAN_STD[1]),
            Normalize(-Camelyon17.MEAN_STD[0], 1),
        ]
    )
    # Convert (B,C,H,W) Torch Tensor to (B,H,W,C) Numpy Array
    imgs = unnorm_transform(imgs)
    imgs = imgs.detach().cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    labels = labels.detach().cpu().numpy()
    batch_size = imgs.shape[0]
    rows = int(np.sqrt(batch_size))
    cols = batch_size // rows
    fig, axs = plt.subplots(rows, cols, figsize=(16, 10))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            img = imgs[idx]
            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            axs[row][col].imshow(img)
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
    plt.tight_layout()
    plt.show()


# Overwrite callbacks
class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str, name: str = "out"):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.name = name

    def write_on_batch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        torch.save(prediction, os.path.join(self.output_dir, dataloader_idx, f"{batch_idx}.pt"))

    def write_on_epoch_end(
        self, trainer, pl_module: "LightningModule", predictions: dict[Any], batch_indices: list[Any] = []
    ):
        predictions = transform_batch_outputs(predictions)
        d_idcs = list(predictions.keys())

        # Save the predictions for each dataset separately.
        for d_idx in d_idcs:
            # We can construct the name from a) The dataset name attribute
            #                                b) The pl_module.metric_suffix attribute
            #                                c) the d_idx
            name = None
            # if hasattr(trainer, "test_dataloaders") and len(trainer.test_dataloaders) == max(d_idcs) + 1:
            #     if trainer.test_dataloaders[d_idx].dataset.name is not None:
            #         name = trainer.test_dataloaders[d_idx].dataset.name

            # Option b) and c)
            if name is None:
                name_suffix = construct_metric_suffix(pl_module, d_idx, "test")
                if name_suffix == "":
                    name = "test"
                else:
                    name = "test_" + name_suffix[1:]

            torch.save(predictions[d_idx], os.path.join(self.output_dir, f"{name}_{self.name}.preds"))

    def on_test_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        predictions = pl_module.predictions
        # epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        self.write_on_epoch_end(trainer, pl_module, predictions)

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Sequence[Any]) -> None:
        return
        # predictions = pl_module.predictions
        # epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        # self.write_on_epoch_end(trainer, pl_module, predictions, batch_indices=[])


def initialize_torchmetrics(self, num_classes, max_num_datasets):

    # Used everywhere
    self.accuracy = nn.ModuleDict(
        {
            step: nn.ModuleList([Accuracy() for _ in range(max_num_datasets)])
            for step in ["train_metrics", "val_metrics", "test_metrics"]
        }
    )
    self.b_accuracy = nn.ModuleDict(
        {
            step: nn.ModuleList([Accuracy(num_classes=num_classes, average="macro") for _ in range(max_num_datasets)])
            for step in ["train_metrics", "val_metrics", "test_metrics"]
        }
    )
    self.conf_matrix = nn.ModuleDict(
        {
            step: nn.ModuleList(
                [ConfusionMatrix(num_classes=num_classes, normalize=None) for _ in range(max_num_datasets)]
            )
            for step in ["train_metrics", "val_metrics", "test_metrics"]
        }
    )
    self.f1_score = nn.ModuleDict(
        # Compute F1 Score only for tumor class
        {
            step: nn.ModuleList([F1Score(num_classes, ignore_index=0) for _ in range(max_num_datasets)])
            for step in ["train_metrics", "val_metrics", "test_metrics"]
        }
    )

    # Only in val and test
    self.auroc = nn.ModuleDict(
        # Compute AUROC only for tumor class
        {
            step: nn.ModuleList([AUROC(num_classes=None, pos_label=1, average=None) for _ in range(max_num_datasets)])
            for step in ["val_metrics", "test_metrics"]
        }
    )

    self.avg_prec = nn.ModuleDict(
        # Compute AvgPREC Score only for tumor class
        {
            step: nn.ModuleList(
                [AveragePrecision(num_classes=None, pos_label=1, average=None) for _ in range(max_num_datasets)]
            )
            for step in ["val_metrics", "test_metrics"]
        }
    )

    self.ece = nn.ModuleDict(
        {
            step: nn.ModuleList([CalibrationError(n_bins=20, norm="l1") for _ in range(max_num_datasets)])
            for step in ["val_metrics", "test_metrics"]
        }
    )

    # self.ace = nn.ModuleDict(
    #     {
    #         step: nn.ModuleList([AdaptiveCalibrationError() for _ in range(max_num_datasets)])
    #         for step in ["val_metrics", "test_metrics"]
    #     }
    # )


def log_conf_matrix(self, name: str, conf_matrix: torch.Tensor):
    df_cm = pd.DataFrame(
        conf_matrix.detach().cpu().numpy(),
        index=np.arange(self.num_classes),
        columns=np.arange(self.num_classes),
    )
    sns.set(font_scale=1.2)
    fig = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Oranges").get_figure()
    # plt.close(fig)  # Done in add_figure
    self.loggers[0].experiment.add_figure(  # Assume tensorboard_logger is always at index 0
        name, fig, self.current_epoch
    )


def construct_metric_suffix(l_model: LightningModule, d_idx: int, split: str) -> str:

    metric_suffix = None
    if hasattr(l_model, "metric_suffix"):
        metric_suffixes = l_model.metric_suffix
        if isinstance(metric_suffixes, int) or isinstance(metric_suffixes, str):
            metric_suffix = metric_suffixes
        elif isinstance(metric_suffixes, dict):
            if split in metric_suffixes:
                metric_suffixes = metric_suffixes[split]

        if isinstance(metric_suffixes, dict):
            metric_suffix = metric_suffixes.get(d_idx, None)
        elif isinstance(metric_suffixes, list):
            metric_suffix = metric_suffixes[d_idx]

    if metric_suffix is None:
        metric_suffix = str(d_idx) if d_idx != 0 else ""
    elif metric_suffix == "":
        pass
    else:
        metric_suffix = "_" + metric_suffix

    return metric_suffix


def log_train_metrics(l_model, d_idx, preds, y, loss):

    metric_suffix = construct_metric_suffix(l_model, d_idx, "train")

    l_model.accuracy["train_metrics"][d_idx](preds, y)
    l_model.b_accuracy["train_metrics"][d_idx](preds, y)
    l_model.conf_matrix["train_metrics"][d_idx].update(preds, y)
    l_model.f1_score["train_metrics"][d_idx](preds, y)

    l_model.log("train_loss" + metric_suffix, loss)
    l_model.log(
        "train_acc" + metric_suffix,
        l_model.accuracy["train_metrics"][d_idx],
        on_step=True,
        on_epoch=True,
        prog_bar=False,
    )
    l_model.log(
        "train_b_acc" + metric_suffix,
        l_model.b_accuracy["train_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )
    l_model.log(
        "train_f1" + metric_suffix,
        l_model.f1_score["train_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )


def log_val_metrics(l_model, d_idx, preds, y, out_sm, loss):

    metric_suffix = construct_metric_suffix(l_model, d_idx, "val")

    l_model.accuracy["val_metrics"][d_idx](preds, y)
    l_model.b_accuracy["val_metrics"][d_idx](preds, y)
    l_model.conf_matrix["val_metrics"][d_idx].update(preds, y)
    l_model.f1_score["val_metrics"][d_idx](preds, y)
    l_model.auroc["val_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label
    l_model.avg_prec["val_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label

    l_model.log("val_loss" + metric_suffix, loss, on_step=True)
    l_model.log(
        "val_acc" + metric_suffix, l_model.accuracy["val_metrics"][d_idx], on_step=True, on_epoch=True, prog_bar=False
    )
    l_model.log(
        "val_b_acc" + metric_suffix,
        l_model.b_accuracy["val_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=True,
    )
    l_model.log(
        "val_f1" + metric_suffix, l_model.f1_score["val_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    )
    l_model.log(
        "val_auroc" + metric_suffix, l_model.auroc["val_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    )
    l_model.log(
        "val_avg_prec" + metric_suffix,
        l_model.avg_prec["val_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )


def log_test_metrics(l_model: LightningModule, d_idx, preds, y, out_sm, loss):

    metric_suffix = construct_metric_suffix(l_model, d_idx, "test")

    l_model.accuracy["test_metrics"][d_idx](preds, y)
    l_model.b_accuracy["test_metrics"][d_idx](preds, y)
    l_model.conf_matrix["test_metrics"][d_idx].update(preds, y)
    l_model.f1_score["test_metrics"][d_idx](preds, y)
    l_model.auroc["test_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label
    l_model.avg_prec["test_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label
    # l_model.ece["test_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label
    # l_model.ace["test_metrics"][d_idx](out_sm[:, 1], y)  # Provide tumor probabilities and label

    l_model.log("test_loss" + metric_suffix, loss, on_step=False, on_epoch=True)
    l_model.log(
        "test_acc" + metric_suffix,
        l_model.accuracy["test_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )
    l_model.log(
        "test_b_acc" + metric_suffix,
        l_model.b_accuracy["test_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )
    l_model.log(
        "test_f1" + metric_suffix, l_model.f1_score["test_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    )
    l_model.log(
        "test_auroc" + metric_suffix, l_model.auroc["test_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    )
    l_model.log(
        "test_avg_prec" + metric_suffix,
        l_model.avg_prec["test_metrics"][d_idx],
        on_step=False,
        on_epoch=True,
        prog_bar=False,
    )
    # l_model.log(
    #     "test_ece" + metric_suffix, l_model.ece["test_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    # )
    # l_model.log(
    #     "test_ace" + metric_suffix, l_model.ace["test_metrics"][d_idx], on_step=False, on_epoch=True, prog_bar=False
    # )


class LitTileClassifier(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 2,
        lr: float = 1e-4,
        max_num_datasets: int = 2,
    ):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.save_hyperparameters(ignore="model")
        self.max_num_datasets = max_num_datasets
        self.predictions = []
        initialize_torchmetrics(self, num_classes, max_num_datasets)

    def forward(self, x):
        out = self.model(x)

        return out

    def evaluate(self, batch, loss=True):
        x, y = batch
        out = self.forward(x)
        if isinstance(out, list):
            out = out[0]

        out_sm = F.softmax(out, dim=1)
        conf, preds = torch.max(out_sm, dim=1)

        if loss:
            loss = F.cross_entropy(out, y)
        else:
            loss = None

        return loss, out_sm, preds, y

    def training_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, out_sm, preds, y = self.evaluate(batch)
        if DEBUG:
            img, y = batch
            n_tumor = torch.sum(y == 1)
            print("Tumor Tiles in Batch: ", n_tumor.item())
            display_batch(batch)
            train_acc = torch.sum(preds == y)
            train_acc = train_acc / len(preds)
            print("Train Acc: ", train_acc.item())
        log_train_metrics(self, d_idx, preds, y, loss)
        return {"d_idx": d_idx, "loss": loss}

    def training_epoch_end(self, outs):

        if not isinstance(outs[0], list):
            outs = [outs]

        d_idcs = list(range(len(outs)))

        for d_idx in d_idcs:
            metric_suffix = construct_metric_suffix(self, d_idx, "train")
            log_conf_matrix(
                self, "train_conf_matrix_epoch" + metric_suffix, self.conf_matrix["train_metrics"][d_idx].compute()
            )
            self.conf_matrix["train_metrics"][d_idx].reset()

    def validation_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, out_sm, preds, y = self.evaluate(batch)
        log_val_metrics(self, d_idx, preds, y, out_sm, loss)
        return {"d_idx": d_idx, "loss": loss}

    def validation_epoch_end(self, outs):

        if not isinstance(outs[0], list):
            outs = [outs]

        d_idcs = list(range(len(outs)))

        for d_idx in d_idcs:
            metric_suffix = construct_metric_suffix(self, d_idx, "val")
            log_conf_matrix(
                self, "val_conf_matrix_epoch" + metric_suffix, self.conf_matrix["val_metrics"][d_idx].compute()
            )
            self.conf_matrix["val_metrics"][d_idx].reset()

    def test_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, out_sm, preds, y = self.evaluate(batch)
        log_test_metrics(self, d_idx, preds, y, out_sm, loss)
        return {"d_idx": d_idx, "softmax": out_sm, "label": y}

    def test_epoch_end(self, outs):

        self.predictions = outs

        # if not isinstance(outs[0], list):
        #     d_idcs = [0]
        # else:
        #     d_idcs = list(range(len(outs)))

        # for d_idx in d_idcs:
        #     metric_suffix = construct_metric_suffix(self, d_idx, "test")

        #     log_conf_matrix(
        #         self, "test_conf_matrix_data_" + metric_suffix, self.conf_matrix["test_metrics"][d_idx].compute()
        #     )
        #     self.conf_matrix["test_metrics"][d_idx].reset()

        #     fig, ax = self.ece["test_metrics"][d_idx].plot_reliability_diagram()
        #     self.loggers[0].experiment.add_figure("ECE Reliability Diagram" + metric_suffix, fig, self.current_epoch)
        #     fig, ax = self.ace["test_metrics"][d_idx].plot_reliability_diagram()
        #     self.loggers[0].experiment.add_figure("AECE Reliability Diagram" + metric_suffix, fig, self.current_epoch)
        #     self.ece["test_metrics"][d_idx].reset()
        #     self.ace["test_metrics"][d_idx].reset()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        loss, out_sm, preds, y = self.evaluate(batch, loss=False)
        # Can't log in predict step

        return {"softmax": out_sm, "label": y}

    def configure_optimizers(self):
        optimizers = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizers, mode=METRIC_MODE[self.hparams["opti_metric"]], patience=self.hparams["patience"]
        )

        lr_schedulers = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": self.hparams["opti_metric"],
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return ([optimizers], [lr_schedulers])


class EnsembleTileClassifier(LitTileClassifier):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_function: Union[
            EnsembleLosses.MultiHeadCrossEntropyLoss,
            EnsembleLosses.MulHCELossKernelWeightCos,
            EnsembleLosses.MulHCELossOrthoActi,
        ],
        num_classes: int = 2,
        lr: float = 1e-4,
        max_num_datasets: int = 2,
    ):
        super().__init__(model, num_classes, lr, max_num_datasets)

        self.loss_function = loss_function

        # Link here
        if isinstance(loss_function, EnsembleLosses.MulHCELossKernelWeightCos):
            loss_function.weight_func = model.getEnsembleParams
        if isinstance(loss_function, EnsembleLosses.MulHCELossOrthoActi):
            loss_function.acti_func = model.getActivations

    def evaluate(self, batch, loss=True):
        x, y = batch
        out = self.forward(x)

        if not isinstance(out, list):
            out = [out]

        # torch.stack(out) has shape (ensemble_size, batch_size, num_classes)
        ens_sm = F.softmax(torch.stack(out), dim=2)
        mean_sm = torch.mean(ens_sm, dim=0)
        # mean_sm has shape (batch_size, num_classes)
        conf, preds = torch.max(mean_sm, dim=1)

        if loss:
            loss = self.loss_function(out, mean_sm, y)
        else:
            loss = None

        return loss, out, mean_sm, preds, y

    def training_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, ens_out, out_sm, preds, y = self.evaluate(batch)
        log_train_metrics(self, d_idx, preds, y, loss)
        return {"d_idx": d_idx, "loss": loss}

    def validation_step(self, batch, batch_idx: int, d_idx: int = 0):
        loss, ens_out, out_sm, preds, y = self.evaluate(batch)
        log_val_metrics(self, d_idx, preds, y, out_sm, loss)
        return {"d_idx": d_idx, "loss": loss}

    def test_step(self, batch, batch_idx, d_idx=0):
        loss, ens_out, out_sm, preds, y = self.evaluate(batch)
        ens_sm = F.softmax(torch.stack(ens_out), dim=2)
        var_sm, mean_sm = torch.var_mean(ens_sm, dim=0)
        log_test_metrics(self, d_idx, preds, y, out_sm, loss)
        return {"d_idx": d_idx, "softmax": mean_sm, "label": y, "variance": var_sm}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        loss, ens_out, out_sm, preds, y = self.evaluate(batch, loss=False)
        ens_sm = F.softmax(torch.stack(ens_out), dim=2)
        var_sm, mean_sm = torch.var_mean(ens_sm, dim=0)
        return {"softmax": mean_sm, "label": y, "variance": var_sm}

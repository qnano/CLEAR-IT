"""HDF5-capable MAPS trainer vendored into CLEAR-IT for public benchmarking.

This file is adapted from the MAPS project:
https://github.com/mahmoodlab/MAPS

MAPS paper:
Shaban, M., Bai, Y., Qiu, H. et al. MAPS: pathologist-level cell type
annotation from tissue images through machine learning. Nat Commun 15, 28
(2024). https://doi.org/10.1038/s41467-023-44188-w

The MAPS repository is distributed under Apache 2.0 with Commons Clause for
non-commercial academic use. See ``clearit/maps_benchmark/THIRD_PARTY_NOTICES.md``
for attribution and license details for the vendored MAPS-derived components.

CLEAR-IT-specific changes in this file:
- reads canonical CLEAR-IT embedding HDF5 files instead of MAPS CSV inputs
- supports selecting one or more HDF5 data parts (``expressions``, ``features``)
- keeps the benchmark runnable from the public CLEAR-IT repository
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .datasets import CellExpressionDataset
from .networks import MLP


class Trainer:
    """Train and evaluate the MAPS MLP on CLEAR-IT embedding HDF5 files."""

    def __init__(
        self,
        *,
        model_checkpoint_path: str | None = None,
        results_dir: str = "./results/",
        data_parts: list[str] | tuple[str, ...] = ("expressions",),
        hidden_dim: int = 512,
        network_depth: int = 4,
        num_features: int = 47,
        num_classes: int = 12,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        dropout: float = 0.10,
        max_epochs: int = 500,
        min_epochs: int = 250,
        patience: int = 100,
        seed: int = 7325111,
        num_workers: int = 4,
        norm: bool = True,
        verbose: int = 0,
    ) -> None:
        self.model_checkpoint_path = model_checkpoint_path
        self.results_dir = str(results_dir)
        self.data_parts = list(data_parts)
        self.hidden_dim = hidden_dim
        self.network_depth = network_depth
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.num_workers = num_workers
        self.seed = seed
        self.norm = norm
        self.verbose = verbose

        self.model: MLP | None = None
        self.optimizer: optim.Optimizer | None = None
        self.loss_fn: nn.Module | None = None
        self.counter = 0
        self.lowest_loss = np.inf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_seed(self) -> None:
        random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def init_model(self) -> None:
        self.model = MLP(
            input_dim=self.num_features,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=self.dropout,
            network_depth=self.network_depth,
        )
        self.model.to(self.device, dtype=torch.float64)

    def init_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("Model must be initialized before the optimizer")
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
        )

    def init_loss_function(self) -> None:
        self.loss_fn = nn.CrossEntropyLoss()

    def save_model(self, mean: np.ndarray | None, std: np.ndarray | None) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Cannot save an uninitialized model")
        save_dict = {
            "model_parameters": self.model.state_dict(),
            "train_data_mean": mean,
            "train_data_std": std,
        }
        ckpt_path = Path(self.results_dir) / "best_checkpoint.pt"
        torch.save(save_dict, ckpt_path)
        self.model_checkpoint_path = str(ckpt_path)

    def load_model(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.model_checkpoint_path is None:
            raise ValueError("Model checkpoint path is None.")
        if self.model is None:
            self.init_model()
        loaded_dict = torch.load(self.model_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(loaded_dict["model_parameters"], strict=True)
        return loaded_dict.get("train_data_mean"), loaded_dict.get("train_data_std")

    def fit(
        self,
        hdf5_path: str | Path,
        group_ids_train: list[str] | None = None,
        group_ids_val: list[str] | None = None,
    ) -> None:
        self.counter = 0
        self.lowest_loss = np.inf
        self.set_seed()
        self.init_model()
        self.init_optimizer()
        self.init_loss_function()

        X_train, y_train, _ = self._load_groups(hdf5_path, group_ids_train)
        if group_ids_val is not None:
            X_val, y_val, _ = self._load_groups(hdf5_path, group_ids_val)
        else:
            idx = np.arange(len(y_train))
            tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=self.seed)
            X_val, y_val = X_train[va_idx], y_train[va_idx]
            X_train, y_train = X_train[tr_idx], y_train[tr_idx]

        train_dataset = CellExpressionDataset(X_train, y_train, is_train=True, norm=self.norm)
        mean, std = train_dataset.mean, train_dataset.std
        valid_dataset = CellExpressionDataset(
            X_val,
            y_val,
            is_train=False,
            mean=mean,
            std=std,
            norm=self.norm,
        )

        train_dl = CellExpressionDataset.get_data_loader(
            train_dataset,
            batch_size=self.batch_size,
            is_train=True,
            num_workers=self.num_workers,
        )
        valid_dl = CellExpressionDataset.get_data_loader(
            valid_dataset,
            batch_size=self.batch_size,
            is_train=False,
            num_workers=self.num_workers,
        )

        result_dict = {
            "train_loss": [],
            "valid_loss": [],
            "train_acc": [],
            "valid_acc": [],
            "train_auc": [],
            "valid_auc": [],
        }
        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_loss, train_acc, train_auc = self.train_loop(train_dl)
            print(
                "\rTrain Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, train_auc: {}                 ".format(
                    epoch,
                    train_loss,
                    train_acc,
                    _fmt_metric(train_auc),
                )
            )
            result_dict["train_loss"].append(train_loss)
            result_dict["train_acc"].append(train_acc)
            result_dict["train_auc"].append(train_auc)

            valid_loss, valid_acc, valid_auc = self.valid_loop(valid_dl)
            print(
                "\rValid Epoch: {}, valid_loss: {:.4f}, valid_acc: {:.4f}, valid_auc: {}                 ".format(
                    epoch,
                    valid_loss,
                    valid_acc,
                    _fmt_metric(valid_auc),
                )
            )
            result_dict["valid_loss"].append(valid_loss)
            result_dict["valid_acc"].append(valid_acc)
            result_dict["valid_auc"].append(valid_auc)

            if self.lowest_loss > valid_loss:
                print("--------------------Saving best model--------------------")
                self.save_model(mean, std)
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print(f"Loss is not decreased in last {self.counter} epochs")

            if (self.counter > self.patience) and (epoch >= self.min_epochs):
                break

            total_time = time.time() - start_time
            print(f"Time to process epoch({epoch}): {total_time / 60:.4f} minutes\n")
            pd.DataFrame.from_dict(result_dict).to_csv(
                Path(self.results_dir) / "training_logs.csv",
                index=False,
            )

    def predict(
        self,
        hdf5_path: str | Path,
        group_ids: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.set_seed()
        self.init_model()
        train_data_mean, train_data_std = self.load_model()

        X, labels, row_group_ids, coords = self._load_groups(
            hdf5_path,
            group_ids,
            return_coords=True,
        )
        test_dataset = CellExpressionDataset(
            X,
            labels,
            is_train=False,
            mean=train_data_mean,
            std=train_data_std,
            norm=self.norm,
        )
        data_loader = CellExpressionDataset.get_data_loader(
            test_dataset,
            batch_size=self.batch_size,
            is_train=False,
            num_workers=self.num_workers,
        )
        pred_labels, pred_probs = self.test_loop(data_loader)
        return pred_labels, pred_probs, labels, coords, row_group_ids

    def _load_groups(
        self,
        hdf5_path: str | Path,
        group_ids: list[str] | None,
        *,
        return_coords: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_parts = {part: [] for part in self.data_parts}
        y_list: list[np.ndarray] = []
        row_group_ids: list[np.ndarray] = []
        coords_list: list[np.ndarray] = []

        with h5py.File(hdf5_path, "r") as handle:
            selected_groups = list(handle.keys()) if group_ids is None else [str(g) for g in group_ids]
            for group_id in tqdm(selected_groups, desc="Loading HDF5 groups"):
                grp = handle[str(group_id)]
                missing = [part for part in self.data_parts + ["labels"] if part not in grp]
                if missing:
                    raise KeyError(f"Group {group_id} is missing datasets: {missing}")

                part_mats = []
                for part in self.data_parts:
                    arr = grp[part][:]
                    x_parts[part].append(arr)
                    part_mats.append(arr)
                labels = grp["labels"][:]
                y_list.append(labels)
                row_group_ids.append(np.array([str(group_id)] * len(labels), dtype=object))

                if return_coords:
                    if "coords" in grp:
                        coords = grp["coords"][:]
                    elif "cell_x" in grp and "cell_y" in grp:
                        coords = np.concatenate([grp["cell_x"][:], grp["cell_y"][:]], axis=1)
                    else:
                        coords = np.empty((len(labels), 0), dtype=np.float32)
                    coords_list.append(coords)

        stacked = [np.concatenate(x_parts[part], axis=0) for part in self.data_parts]
        X = stacked[0] if len(stacked) == 1 else np.concatenate(stacked, axis=1)
        y = np.concatenate(y_list, axis=0)
        group_array = np.concatenate(row_group_ids, axis=0)
        if return_coords:
            coords_array = np.concatenate(coords_list, axis=0) if coords_list else np.empty((len(y), 0), dtype=np.float32)
            return X, y, group_array, coords_array
        return X, y, group_array

    def train_loop(self, data_loader):
        if self.model is None or self.optimizer is None or self.loss_fn is None:
            raise RuntimeError("Trainer is not initialized")
        total_loss = 0.0
        gt_labels: list[int] = []
        pred_labels: list[int] = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.train()
        for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
            gt_labels.extend(label_batch.cpu().numpy().tolist())
            features_batch = features_batch.to(self.device, dtype=torch.float64)
            label_batch = label_batch.to(self.device)
            logits, probs = self.model(features_batch)
            self.optimizer.zero_grad()
            loss = self.loss_fn(logits, label_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().item()

            p_labels = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
            p_probs = probs.detach().cpu().numpy()
            if pred_probs is None:
                pred_probs = p_probs
            else:
                pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
            pred_labels.extend(p_labels)
            sys.stdout.write(
                "\rTraining Batch {}/{}, avg loss: {:.4f}".format(
                    batch_idx + 1,
                    batch_count,
                    total_loss / (batch_idx + 1),
                )
            )

        acc = accuracy_score(gt_labels, pred_labels)
        auc = _safe_roc_auc_score(gt_labels, pred_probs, self.num_classes)
        if self.verbose == 1:
            print("\nConfusion Matrix")
            conf = pd.DataFrame(
                confusion_matrix(gt_labels, pred_labels, labels=list(range(self.num_classes))),
                index=[f"class_{i}" for i in range(self.num_classes)],
                columns=[f"class_{i}" for i in range(self.num_classes)],
            )
            print(conf)
        return total_loss / max(1, batch_count), acc, auc

    def valid_loop(self, data_loader):
        if self.model is None or self.loss_fn is None:
            raise RuntimeError("Trainer is not initialized")
        total_loss = 0.0
        gt_labels: list[int] = []
        pred_labels: list[int] = []
        pred_probs = None
        batch_count = len(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (features_batch, label_batch) in enumerate(data_loader):
                gt_labels.extend(label_batch.cpu().numpy().tolist())
                features_batch = features_batch.to(self.device, dtype=torch.float64)
                label_batch = label_batch.to(self.device)
                logits, probs = self.model(features_batch)
                loss = self.loss_fn(logits, label_batch)
                total_loss += loss.cpu().item()

                p_labels = torch.argmax(probs, dim=1).detach().cpu().numpy().tolist()
                p_probs = probs.detach().cpu().numpy()
                if pred_probs is None:
                    pred_probs = p_probs
                else:
                    pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                pred_labels.extend(p_labels)
                sys.stdout.write(
                    "\rValidation Batch {}/{}, avg loss: {:.4f}".format(
                        batch_idx + 1,
                        batch_count,
                        total_loss / (batch_idx + 1),
                    )
                )

        acc = accuracy_score(gt_labels, pred_labels)
        auc = _safe_roc_auc_score(gt_labels, pred_probs, self.num_classes)
        if self.verbose == 1:
            print("\nConfusion Matrix")
            conf = pd.DataFrame(
                confusion_matrix(gt_labels, pred_labels, labels=list(range(self.num_classes))),
                index=[f"class_{i}" for i in range(self.num_classes)],
                columns=[f"class_{i}" for i in range(self.num_classes)],
            )
            print(conf)
        return total_loss / max(1, batch_count), acc, auc

    def test_loop(self, data_loader):
        if self.model is None:
            raise RuntimeError("Trainer is not initialized")
        pred_labels = None
        pred_probs = None
        batch_count = len(data_loader)
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (features_batch, _) in enumerate(data_loader):
                features_batch = features_batch.to(self.device, dtype=torch.float64)
                _, probs = self.model(features_batch)
                p_labels = torch.argmax(probs, dim=1).detach().cpu().numpy()
                p_probs = probs.detach().cpu().numpy()
                if pred_labels is None:
                    pred_labels = p_labels
                    pred_probs = p_probs
                else:
                    pred_labels = np.concatenate((pred_labels, p_labels), axis=0)
                    pred_probs = np.concatenate((pred_probs, p_probs), axis=0)
                sys.stdout.write(f"\rTesting Batch {batch_idx + 1}/{batch_count}")
        return pred_labels, pred_probs


def _safe_roc_auc_score(gt_labels, pred_probs, num_classes):
    if pred_probs is None:
        return np.nan
    try:
        return roc_auc_score(
            gt_labels,
            pred_probs,
            multi_class="ovo",
            labels=[i for i in range(num_classes)],
        )
    except ValueError:
        return np.nan


def _fmt_metric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.4f}"

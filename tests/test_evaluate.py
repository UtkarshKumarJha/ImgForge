import csv

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate import tune_threshold, save_predictions, predict, is_degenerate
from experiment_log import LOG_COLS, append_to_log


def test_tune_threshold_maximises_macro_f1_on_validation():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.55, 0.60, 0.65, 0.70, 0.80])
    t = tune_threshold(y, p)
    from sklearn.metrics import f1_score
    achieved = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
    assert achieved == pytest.approx(1.0)


def test_save_predictions_roundtrip(tmp_path):
    f = tmp_path / "r.npz"
    save_predictions(str(f), ["a.jpg", "b.jpg"], np.array([0, 1]), np.array([0.2, 0.8]))
    d = np.load(f, allow_pickle=False)
    assert list(d["paths"]) == ["a.jpg", "b.jpg"]
    assert d["labels"].tolist() == [0, 1]
    assert d["probs"].tolist() == pytest.approx([0.2, 0.8])


def test_degenerate_predictions_are_flagged():
    assert is_degenerate(np.array([0, 0, 0, 0]))
    assert not is_degenerate(np.array([0, 0, 1, 1]))


class _FlipSensitiveModel(torch.nn.Module):
    """Logit depends on the first column, so a horizontal flip changes it."""

    def forward(self, x):
        v = x[:, :, :, 0].mean(dim=(1, 2))
        return torch.stack([-v, v], dim=1)


class _OneImageDataset(Dataset):
    def __init__(self):
        self.samples = [("a.jpg", 1)]
        self.image = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.image, torch.tensor(1)


def test_tta_averages_image_and_horizontal_flip():
    loader = DataLoader(_OneImageDataset(), batch_size=1)
    model = _FlipSensitiveModel()
    y_true, probs_plain, paths = predict(model, loader, "cpu", tta=False)
    _, probs_tta, _ = predict(model, loader, "cpu", tta=True)

    p_image = torch.sigmoid(torch.tensor(2.0)).item()
    p_flip = 0.5
    assert probs_plain.tolist() == pytest.approx([p_image])
    assert probs_tta.tolist() == pytest.approx([(p_image + p_flip) / 2])
    assert y_true.tolist() == [1]
    assert paths == ["a.jpg"]


def test_log_accepts_result_without_new_columns(tmp_path):
    legacy = {
        "checkpoint": "c.pth", "arch": "mlp", "input_mode": "rgb", "seed": 1,
        "split": "test", "n_samples": 2,
        "metrics": {"accuracy": {"value": 1.0, "ci_lo": 1.0, "ci_hi": 1.0}},
    }
    log = tmp_path / "log.csv"
    append_to_log(legacy, str(log))

    with open(log, newline="") as f:
        row = next(csv.DictReader(f))
    assert list(row) == LOG_COLS
    assert row["f1_macro_tuned"] == ""
    assert row["degenerate"] == ""

import numpy as np

from scripts.final_table import ensemble, val_f1


def _npz(path, probs, threshold):
    np.savez(path, paths=np.array(["a.jpg", "b.tif", "c.jpg", "d.jpg"]),
             labels=np.array([0, 1, 1, 0]), probs=np.array(probs), threshold=threshold)
    return str(path)


def test_ensemble_averages_probabilities_and_thresholds(tmp_path):
    e = ensemble([_npz(tmp_path / "s1.npz", [0.1, 0.9, 0.2, 0.6], 0.5),
                  _npz(tmp_path / "s2.npz", [0.3, 0.7, 0.8, 0.4], 0.3)])
    # mean probs [0.2, 0.8, 0.5, 0.5] cut at 0.4 -> [0, 1, 1, 1]: one false positive
    assert abs(e["threshold"] - 0.4) < 1e-12
    assert e["accuracy"] == 0.75


def test_val_f1_reads_the_last_reported_best(tmp_path):
    (tmp_path / "r.log").write_text("Best val F1: 0.70\n...\nTraining complete. Best val F1: 0.8123\n")
    assert val_f1(str(tmp_path), "r") == 0.8123
    assert val_f1(str(tmp_path), "missing") is None

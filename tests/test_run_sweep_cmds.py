import shutil

import run_sweep
from run_sweep import build_eval_cmd, build_train_cmd, persist_to_drive

RUN = "resnet18_bilstm_rgb_ela_s1"


class A:
    data_dir = "data"; num_epochs = 60; batch_size = 16; num_workers = 2
    ela_quality = 75; out_dir = "models"; wandb = False
    image_size = 256; lr_new = 1e-3; warmup = 5
    grad_clip = 1.0; fusion = "dual"; tta = True


def test_train_cmd_carries_every_new_flag():
    cmd = build_train_cmd(A, "swin_tiny", "rgb_ela", 2)
    for flag, val in [("--image-size", "256"), ("--lr-new", "0.001"),
                      ("--warmup", "5"), ("--grad-clip", "1.0"),
                      ("--fusion", "dual")]:
        assert flag in cmd and cmd[cmd.index(flag) + 1] == val, flag
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "2"


def test_train_cmd_leaves_backbone_lr_to_the_arch():
    assert "--lr-backbone" not in build_train_cmd(A, "swin_tiny", "rgb_ela", 2)


def test_eval_cmd_enables_tta_and_json_output():
    cmd = build_eval_cmd(A, "models/m.pth", "results/m.json")
    assert "--tta" in cmd
    assert cmd[cmd.index("--out-json") + 1] == "results/m.json"


def test_eval_cmd_omits_tta_when_disabled():
    class NoTTA(A):
        tta = False

    assert "--tta" not in build_eval_cmd(NoTTA, "m.pth", "m.json")


def test_persist_copies_predictions(tmp_path):
    preds = tmp_path / "predictions"
    preds.mkdir()
    (preds / f"{RUN}_test.npz").write_bytes(b"x")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")
    drive = tmp_path / "drive"

    persist_to_drive(str(log), None, str(drive), RUN, str(preds))

    assert (drive / "predictions" / f"{RUN}_test.npz").read_bytes() == b"x"


def test_persist_defaults_to_predictions_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / f"{RUN}_test.npz").write_bytes(b"x")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")

    persist_to_drive(str(log), None, str(tmp_path / "drive"), RUN)

    assert (tmp_path / "drive" / "predictions" / f"{RUN}_test.npz").read_bytes() == b"x"


def test_persist_warns_when_predictions_dir_missing(tmp_path, capsys):
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")
    drive = tmp_path / "drive"

    persist_to_drive(str(log), None, str(drive), RUN, str(tmp_path / "nope"))

    out = capsys.readouterr().out
    assert "WARNING" in out and RUN in out and "nope" in out
    assert (drive / "experiment_log.csv").exists()


def test_persist_warns_when_run_npz_missing(tmp_path, capsys):
    preds = tmp_path / "predictions"
    preds.mkdir()
    (preds / "some_other_run_test.npz").write_bytes(b"x")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")

    persist_to_drive(str(log), None, str(tmp_path / "drive"), RUN, str(preds))

    out = capsys.readouterr().out
    assert "WARNING" in out and RUN in out and str(preds) in out


def test_persist_writes_log_last(tmp_path, monkeypatch):
    preds = tmp_path / "predictions"
    preds.mkdir()
    (preds / f"{RUN}_test.npz").write_bytes(b"x")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "img.npy").write_bytes(b"x")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")
    drive = tmp_path / "drive"

    copied = []
    real_copy2 = shutil.copy2

    def record(src, dst):
        copied.append(dst)
        real_copy2(src, dst)

    monkeypatch.setattr(run_sweep.shutil, "copy2", record)

    persist_to_drive(str(log), str(cache), str(drive), RUN, str(preds))

    assert copied[-1] == str(drive / "experiment_log.csv")
    assert str(drive / "predictions" / f"{RUN}_test.npz") in copied

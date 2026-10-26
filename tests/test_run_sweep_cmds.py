import shutil

import run_sweep
from run_sweep import (
    REFERENCE, build_eval_cmd, build_train_cmd, load_completed, persist_to_drive,
    restore_from_drive, run_name_for,
)

RUN = "resnet18_bilstm_rgb_ela_s1"


class A:
    """The declared sweep config: every field tracked from REFERENCE."""
    data_dir = "data"; num_workers = 2; out_dir = "models"; wandb = False
    ela_quality = REFERENCE["ela_quality"]; batch_size = REFERENCE["batch_size"]
    image_size = REFERENCE["image_size"]; num_epochs = REFERENCE["num_epochs"]
    lr_new = REFERENCE["lr_new"]; warmup = REFERENCE["warmup"]
    grad_clip = REFERENCE["grad_clip"]; fusion = REFERENCE["fusion"]; tta = True


def test_train_cmd_carries_every_new_flag():
    cmd = build_train_cmd(A, "swin_tiny", "rgb_ela", 2, RUN)
    for flag, val in [("--image-size", str(REFERENCE["image_size"])),
                      ("--lr-new", "0.001"), ("--warmup", "5"),
                      ("--grad-clip", "1.0"), ("--fusion", "dual")]:
        assert flag in cmd and cmd[cmd.index(flag) + 1] == val, flag
    assert "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "2"
    # the sweep's deviation-suffixed name is what it looks the checkpoint up by
    assert cmd[cmd.index("--run-name") + 1] == RUN


def test_train_cmd_leaves_backbone_lr_to_the_arch():
    assert "--lr-backbone" not in build_train_cmd(A, "swin_tiny", "rgb_ela", 2, RUN)


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


def test_run_name_has_no_suffix_for_the_reference_config():
    assert run_name_for("swin_tiny", "rgb_ela", 2, A) == "swin_tiny_rgb_ela_s2"


def test_deviating_config_changes_the_run_name():
    class P256(A):
        image_size = 256

    class Concat(A):
        fusion = "concat"

    class Both(P256):
        fusion = "concat"

    class NoTTA(A):
        tta = False

    assert run_name_for("swin_tiny", "rgb_ela", 2, P256) == "swin_tiny_rgb_ela_p256_s2"
    assert run_name_for("swin_tiny", "rgb_ela", 2, Concat) == "swin_tiny_rgb_ela_concat_s2"
    assert run_name_for("swin_tiny", "rgb_ela", 2, Both) == "swin_tiny_rgb_ela_p256_concat_s2"
    assert run_name_for("swin_tiny", "rgb_ela", 2, NoTTA) == "swin_tiny_rgb_ela_notta_s2"


def test_deviation_keeps_seed_last_so_mcnemar_can_parse_it():
    from analysis.mcnemar import parse_run_name

    class P256(A):
        image_size = 256

    name = run_name_for("resnet18_bilstm", "rgb_ela", 3, P256)
    assert parse_run_name(name) == ("resnet18_bilstm", "rgb_ela")
    assert name.rsplit("_s", 1)[1] == "3"


def test_load_completed_uses_the_run_name_column(tmp_path):
    log = tmp_path / "log.csv"
    log.write_text("arch,input_mode,seed,run_name\n"
                   "swin_tiny,rgb_ela,1,swin_tiny_rgb_ela_p224_s1\n"
                   "swin_tiny,rgb_ela,2,\n")   # legacy row: no run_name recorded

    completed = load_completed(str(log))
    assert "swin_tiny_rgb_ela_p224_s1" in completed        # pilot does not block 256
    assert "swin_tiny_rgb_ela_s1" not in completed
    assert "swin_tiny_rgb_ela_s2" in completed             # undeviated fallback


def test_persist_copies_the_checkpoint(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{RUN}_best.pth").write_bytes(b"ckpt")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")

    persist_to_drive(str(log), None, str(tmp_path / "drive"), RUN,
                     str(tmp_path / "predictions"), str(models))

    assert (tmp_path / "drive" / "models" / f"{RUN}_best.pth").read_bytes() == b"ckpt"


def test_persist_warns_when_the_checkpoint_is_missing(tmp_path, capsys):
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")

    persist_to_drive(str(log), None, str(tmp_path / "drive"), RUN,
                     str(tmp_path / "predictions"), str(tmp_path / "models"))

    out = capsys.readouterr().out
    assert "WARNING" in out and "checkpoint" in out


def test_persist_overwrites_a_stale_prediction(tmp_path):
    preds = tmp_path / "predictions"
    preds.mkdir()
    (preds / f"{RUN}_test.npz").write_bytes(b"new")
    drive = tmp_path / "drive"
    (drive / "predictions").mkdir(parents=True)
    (drive / "predictions" / f"{RUN}_test.npz").write_bytes(b"old")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")

    persist_to_drive(str(log), None, str(drive), RUN, str(preds))

    assert (drive / "predictions" / f"{RUN}_test.npz").read_bytes() == b"new"


def test_restore_brings_back_checkpoints_without_clobbering_local_ones(tmp_path):
    drive = tmp_path / "drive"
    (drive / "models").mkdir(parents=True)
    (drive / "models" / "a_best.pth").write_bytes(b"old")
    (drive / "models" / "b_best.pth").write_bytes(b"drive")
    models = tmp_path / "models"
    models.mkdir()
    (models / "a_best.pth").write_bytes(b"local")

    restore_from_drive(str(tmp_path / "log.csv"), None, str(drive), str(models))

    assert (models / "a_best.pth").read_bytes() == b"local"    # local is newer
    assert (models / "b_best.pth").read_bytes() == b"drive"


def test_max_hours_stops_before_starting_another_run(tmp_path, monkeypatch):
    """A capped Kaggle session must stop cleanly instead of being killed mid-run."""
    calls = []
    monkeypatch.setattr(run_sweep, "run_combo", lambda *a, **k: calls.append(a) or True)
    for name in ("restore_from_drive", "warmup_ela_cache", "persist_to_drive"):
        monkeypatch.setattr(run_sweep, name, lambda *a, **k: None)
    monkeypatch.setattr(run_sweep, "load_completed", lambda *a, **k: set())
    clock = iter([0.0, 7200.0])  # start, then two hours later
    monkeypatch.setattr(run_sweep.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr("sys.argv", [
        "run_sweep.py", "--max-hours", "1", "--archs", "mlp", "--input-modes", "rgb",
        "--seeds", "1", "--skip-ela-warmup",
        "--results-dir", str(tmp_path / "results"), "--out-dir", str(tmp_path / "models"),
    ])
    run_sweep.main()
    assert calls == []


def test_gpus_run_concurrently_on_distinct_devices_and_sizes_get_suffixed(tmp_path, monkeypatch):
    import threading
    calls, barrier = [], threading.Barrier(2, timeout=10)

    def fake_run_combo(arch, mode, seed, args, gpu=None, lock=None):
        barrier.wait()  # both GPUs must be busy at the same time
        calls.append((run_name_for(arch, mode, seed, args), gpu))
        return True

    monkeypatch.setattr(run_sweep, "run_combo", fake_run_combo)
    for name in ("restore_from_drive", "warmup_ela_cache", "persist_to_drive"):
        monkeypatch.setattr(run_sweep, name, lambda *a, **k: None)
    monkeypatch.setattr(run_sweep, "load_completed", lambda *a, **k: set())
    monkeypatch.setattr("sys.argv", [
        "run_sweep.py", "--archs", "efficientnet_b0", "--input-modes", "ela",
        "--seeds", "1", "--image-size", "320", "384", "--gpus", "2", "--skip-ela-warmup",
        "--results-dir", str(tmp_path / "results"), "--out-dir", str(tmp_path / "models"),
    ])
    run_sweep.main()
    assert sorted(name for name, _ in calls) == ["efficientnet_b0_ela_p320_s1",
                                                 "efficientnet_b0_ela_p384_s1"]
    assert {gpu for _, gpu in calls} == {0, 1}

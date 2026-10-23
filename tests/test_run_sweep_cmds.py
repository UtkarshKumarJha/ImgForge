from run_sweep import build_eval_cmd, build_train_cmd, persist_to_drive


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
    (preds / "run_test.npz").write_bytes(b"x")
    log = tmp_path / "experiment_log.csv"
    log.write_text("arch\n")
    drive = tmp_path / "drive"

    persist_to_drive(str(log), None, str(drive), str(preds))

    assert (drive / "predictions" / "run_test.npz").read_bytes() == b"x"

import sys

import torch

from model_factory import build_model, build_from_checkpoint
from train import build_param_groups, make_scheduler


def _ids(groups):
    seen = []
    for g in groups:
        seen += [id(p) for p in g["params"]]
    return seen


def test_no_parameter_is_in_two_groups():
    for arch in ["efficientnet_b0", "resnet18_bilstm", "swin_tiny", "mlp"]:
        m = build_model(arch, "rgb_ela")
        ids = _ids(build_param_groups(m, arch, 1e-4, 1e-3))
        assert len(ids) == len(set(ids)), arch


def test_aux_stem_is_in_the_new_group():
    m = build_model("efficientnet_b0", "rgb_ela")
    groups = build_param_groups(m, "efficientnet_b0", 1e-4, 1e-3)
    aux_id = id(m.features[0][0].aux_conv.weight)
    new_ids = [id(p) for g in groups if g["lr"] == 1e-3 for p in g["params"]]
    assert aux_id in new_ids


def test_head_and_lstm_are_in_the_new_group():
    m = build_model("resnet18_bilstm", "rgb_ela")
    groups = build_param_groups(m, "resnet18_bilstm", 1e-4, 1e-3)
    new_ids = {id(p) for g in groups if g["lr"] == 1e-3 for p in g["params"]}
    assert id(m.classifier[1].weight) in new_ids
    assert id(m.lstm.weight_ih_l0) in new_ids
    # The pretrained RGB stem stays on the backbone LR.
    backbone_ids = {id(p) for g in groups if g["lr"] == 1e-4 for p in g["params"]}
    assert id(m.backbone[0].rgb_conv.weight) in backbone_ids


def test_warmup_ramps_then_cosine_decays():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p], "lr": 1e-3}])
    sched = make_scheduler(opt, warmup_epochs=5, total_epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[4]
    assert lrs[5] > lrs[-1]


def test_checkpoint_cfg_round_trips_fusion():
    m = build_model("efficientnet_b0", "rgb_ela", fusion="dual")
    ckpt = {
        "model_state_dict": m.state_dict(),
        "cfg": {"arch": "efficientnet_b0", "input_mode": "rgb_ela", "fusion": "dual"},
    }
    rebuilt = build_from_checkpoint(ckpt)
    rebuilt.load_state_dict(ckpt["model_state_dict"], strict=True)


def test_checkpoint_without_fusion_rebuilds_concat():
    m = build_model("efficientnet_b0", "rgb_ela", fusion="concat")
    ckpt = {
        "model_state_dict": m.state_dict(),
        "cfg": {"arch": "efficientnet_b0", "input_mode": "rgb_ela"},
    }
    rebuilt = build_from_checkpoint(ckpt)
    assert isinstance(rebuilt.features[0][0], torch.nn.Conv2d)
    rebuilt.load_state_dict(ckpt["model_state_dict"], strict=True)


def test_fusion_argument_reaches_cfg(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--fusion", "dual"])
    from train import parse_args
    assert vars(parse_args())["fusion"] == "dual"

import torch
from model_factory import build_model, VALID_ARCHS, VALID_MODES


def test_all_arch_mode_combos_forward():
    for arch in sorted(VALID_ARCHS):
        for mode in sorted(VALID_MODES):
            ch = {"rgb": 3, "ela": 1, "rgb_ela": 4}[mode]
            m = build_model(arch, mode).eval()
            with torch.no_grad():
                out = m(torch.randn(1, ch, 224, 224))
            assert out.shape == (1, 2), (arch, mode, out.shape)


def test_dual_stem_sums_rgb_and_aux_branches():
    import torch.nn as nn
    from model_factory import DualStem
    rgb = nn.Conv2d(3, 8, 3, padding=1)
    aux = nn.Conv2d(1, 8, 3, padding=1)
    x = torch.randn(1, 4, 8, 8)
    expect = rgb(x[:, :3]) + aux(x[:, 3:])
    assert torch.allclose(DualStem(rgb, aux)(x), expect)


def test_ela_only_conv_initialised_from_rgb_mean():
    m = build_model("efficientnet_b0", "ela", fusion="concat")
    from torchvision import models
    ref = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    got = m.features[0][0].weight
    assert got.shape[1] == 1
    assert torch.allclose(got[:, 0], ref.features[0][0].weight.mean(dim=1), atol=1e-6)


def test_build_from_checkpoint_restores_layout():
    from model_factory import build_from_checkpoint, DualStem
    from torchvision import models

    dual_ckpt = {"cfg": {"arch": "efficientnet_b0", "input_mode": "rgb_ela",
                         "fusion": "dual"}}
    m = build_from_checkpoint(dual_ckpt)
    assert isinstance(m.features[0][0], DualStem)
    with torch.no_grad():
        assert m(torch.randn(1, 4, 224, 224)).shape == (1, 2)
    build_from_checkpoint(dual_ckpt).load_state_dict(m.state_dict())

    # No "fusion" key → checkpoints trained before the dual stem use concat.
    concat = build_from_checkpoint({"cfg": {}})
    conv = concat.features[0][0]
    ref = models.efficientnet_b0(weights=None)
    assert conv.in_channels == 4
    assert conv.out_channels == ref.features[0][0].out_channels
    assert not hasattr(conv, "rgb_conv")

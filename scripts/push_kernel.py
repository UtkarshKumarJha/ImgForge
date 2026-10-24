"""Push notebooks/kaggle_sweep.ipynb to Kaggle with a stage override and run it.

Kept in the repo so a resumed session on another machine does not depend on a
scratch directory. The token defaults to ~/.kaggle/access_token; pass
--token-file to drive a second account.

    python3 scripts/push_kernel.py --stage sweep --seeds "1 2 3" --gpu --timeout 43200
    python3 scripts/push_kernel.py --stage analysis --timeout 3600

Splitting one grid across two accounts:

    # account A: the two cheap architectures
    python3 scripts/push_kernel.py --stage sweep --seeds "1 2 3" --gpu \
        --only "--archs efficientnet_b0 resnet18_bilstm mlp" --no-self-source
    # account B: Swin, its own kernel and its own persist chain
    python3 scripts/push_kernel.py --stage sweep --seeds "1 2 3" --gpu \
        --slug theiruser/imgforge-sweep --token-file ~/.kaggle/team_token \
        --only "--archs swin_tiny" --no-self-source
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "kaggle_sweep.ipynb"
DEFAULT_SLUG = "utkarshkumarjha/imgforge-metric-sweep"
CASIA = "divg07/casia-20-image-tampering-detection-dataset"
API = "https://www.kaggle.com/api/v1/kernels/push"


def patch(nb_text: str, stage: str, seeds: str, img: str, only: str) -> str:
    """Override the stage knobs in the first code cell; the repo copy stays at s0."""
    nb = json.loads(nb_text)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if 'STAGE = "s0"' not in src:
            continue
        for key, value in (("STAGE", stage), ("SEEDS", seeds), ("IMG", img), ("ONLY", only)):
            src = re.sub(rf'^{key} = "[^"]*"', f'{key} = "{value}"', src,
                         count=1, flags=re.M)
        cell["source"] = src.splitlines(True)
    return json.dumps(nb, indent=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["s0", "pilot", "sweep", "analysis"])
    p.add_argument("--seeds", default="1")
    p.add_argument("--img", default="")
    p.add_argument("--only", default="", help="extra sweep flags, e.g. \"--archs swin_tiny\"")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--timeout", type=int, default=10800, help="seconds, 43200 max")
    p.add_argument("--slug", default=DEFAULT_SLUG)
    p.add_argument("--token-file", default="~/.kaggle/access_token")
    p.add_argument("--no-self-source", action="store_true",
                   help="start a fresh persist chain instead of mounting this kernel's "
                        "latest output")
    a = p.parse_args()

    token = Path(a.token_file).expanduser().read_text().strip()
    payload = {
        "slug": a.slug,
        "text": patch(NB.read_text(), a.stage, a.seeds, a.img, a.only),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": a.gpu,
        "enableInternet": True,
        "datasetDataSources": [CASIA],
        # The kernel mounts its own latest output, so persist/ survives sessions.
        "kernelDataSources": [] if a.no_self_source else [a.slug],
        "kernelExecutionType": "SaveAndRunAll",
        "sessionTimeoutSeconds": a.timeout,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "imgforge-push/1.0",
                 "Authorization": "Bearer " + token},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    print(f"version {d.get('versionNumber')} | {a.slug} | stage {a.stage} | seeds {a.seeds} "
          f"| gpu {a.gpu} | img {a.img or 'default'} | only {a.only or 'all'} "
          f"| timeout {a.timeout}s")
    print("error:", repr(d.get("error")))
    for key in ("invalidDatasetSources", "invalidKernelSources"):
        if d.get(key):
            print(f"{key}: {d[key]}")


if __name__ == "__main__":
    main()

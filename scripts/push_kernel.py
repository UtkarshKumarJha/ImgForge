"""Push notebooks/kaggle_sweep.ipynb to Kaggle with a stage override and run it.

Kept in the repo so a resumed session on another machine does not depend on a
scratch directory. Reads the API token from ~/.kaggle/access_token.

    python3 scripts/push_kernel.py --stage sweep --seeds "1 2 3" --gpu --timeout 43200
    python3 scripts/push_kernel.py --stage analysis --timeout 3600
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "kaggle_sweep.ipynb"
SLUG = "utkarshkumarjha/imgforge-metric-sweep"
CASIA = "divg07/casia-20-image-tampering-detection-dataset"
API = "https://www.kaggle.com/api/v1/kernels/push"


def patch(nb_text: str, stage: str, seeds: str, img: str) -> str:
    """Override the stage knobs in the first code cell; the repo copy stays at s0."""
    nb = json.loads(nb_text)
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if 'STAGE = "s0"' not in src:
            continue
        src = re.sub(r'^STAGE = "[^"]*"', f'STAGE = "{stage}"', src, count=1, flags=re.M)
        src = re.sub(r'^SEEDS = "[^"]*"', f'SEEDS = "{seeds}"', src, count=1, flags=re.M)
        src = re.sub(r'^IMG = "[^"]*"', f'IMG = "{img}"', src, count=1, flags=re.M)
        cell["source"] = src.splitlines(True)
    return json.dumps(nb, indent=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["s0", "pilot", "sweep", "analysis"])
    p.add_argument("--seeds", default="1")
    p.add_argument("--img", default="")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--timeout", type=int, default=10800, help="seconds, 43200 max")
    p.add_argument("--no-self-source", action="store_true",
                   help="do not mount the previous version's persist/ output")
    a = p.parse_args()

    payload = {
        "slug": SLUG,
        "text": patch(NB.read_text(), a.stage, a.seeds, a.img),
        "language": "python",
        "kernelType": "notebook",
        "isPrivate": True,
        "enableGpu": a.gpu,
        "enableInternet": True,
        "datasetDataSources": [CASIA],
        # The kernel mounts its own latest output, so persist/ survives sessions.
        "kernelDataSources": [] if a.no_self_source else [SLUG],
        "kernelExecutionType": "SaveAndRunAll",
        "sessionTimeoutSeconds": a.timeout,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "imgforge-push/1.0",
                 "Authorization": "Bearer " + (Path.home() / ".kaggle" / "access_token")
                 .read_text().strip()},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    print(f"version {d.get('versionNumber')} | stage {a.stage} | seeds {a.seeds} "
          f"| gpu {a.gpu} | img {a.img or 'default'} | timeout {a.timeout}s")
    print("error:", repr(d.get("error")))
    for key in ("invalidDatasetSources", "invalidKernelSources"):
        if d.get(key):
            print(f"{key}: {d[key]}")


if __name__ == "__main__":
    main()

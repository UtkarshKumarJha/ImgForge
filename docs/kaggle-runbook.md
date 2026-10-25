# Kaggle runbook — metric-improvement sweep

Everything below runs on Kaggle. This machine only pushes versions and reads logs,
so a closed laptop never stops a running session; it only stops the *next* push.

Kernel: https://www.kaggle.com/code/utkarshkumarjha/imgforge-metric-sweep
(version numbers below are what was pushed, newest last)

| version | stage | what it does |
|---|---|---|
| 1–2 | s0 | failed: dataset not attached (API needs `datasetDataSources` in the REST payload) |
| 3 | s0 | dataset attached, but stale 65-row repo CSVs skipped the split |
| 4 | s0 | **real split built**: 11,803 images after 811 near-duplicates |
| 5 | s0 | mount of own output verified, gate green |
| 6 | pilot | 256 arm ok, 224 arm failed on checkpoint naming (fixed) |
| 7 | pilot | rerun with the naming fix + 3-epoch timing probes |
| 8 | sweep | 224 px / 30 epochs, seeds 1 2 3 — **abandoned** when the second account freed the budget for the declared 60-epoch cap |
| 9+ | sweep | **the 36 reported runs**, 224 px / 60 epochs, split across two accounts |

## State

- S0 gate green: `split_ok: true`, `problems: []`, `mask_available: false`
  (CASIA ships no ground-truth masks, so localization coverage will be 0).
- Split, measured: train 9442 (5253 authentic), val 1180 (923), test 1181 (878).
  Fingerprint `79e78eb2…`, recorded in `scripts/verify_setup.py`.
- Declared sweep config (`run_sweep.REFERENCE`): **224 px, 60 epochs** (early stop,
  patience 10), batch 16, dual fusion, lr_new 1e-3, warmup 5. 224 px is the pilot's
  validation winner (0.8104 vs 0.8066 at 256); 60 is the cap the design declares.
  The 30-epoch version run as v8 was a quota concession and is discarded.
- Pilot (v6, 10 epochs, EfficientNet-B0/ELA): best val F1 0.8066 at 256 px,
  0.8104 at 224 px.
- Prediction: 36 runs at 224/60 ≈ 30-40 h of GPU (early stopping cuts most runs
  before 60). Split across two accounts, run concurrently:
  - account A (this one): `efficientnet_b0 resnet18_bilstm mlp` ≈ 18 h
  - account B (borrowed): `swin_tiny` alone ≈ 15-22 h, separate kernel + persist chain

## Resume (any machine with this repo and `~/.kaggle/access_token`)

```bash
python3 scripts/push_kernel.py --stage sweep --seeds "1 2 3" --gpu --timeout 43200
```

The sweep is resumable: completed runs are skipped by `run_name`, and every run
persists `results/` + `models/` + `predictions/` + `data/*.csv` into `persist/`
before the next starts. Re-pushing the same command continues where the last
session stopped, because the kernel mounts its own latest output.

Then, once all 36 rows exist in `persist/results/experiment_log.csv`:

```bash
python3 scripts/push_kernel.py --stage analysis --timeout 3600
```

## S3 pre-flight (done 2026-10-25)

Both analysis CLIs were run end-to-end against a synthetic 36-run set with the
exact canonical names the sweep produces: `analysis.mcnemar` emitted all 12
Family A and 6 Family B tests with Holm applied per family, and
`analysis.confound` completed and reported zero usable quality bins for
unreadable fixture paths instead of raising. The analysis stage therefore
depends only on the 36 real artefacts.

## State at 2026-10-25 05:20 UTC (laptop shut down from here)

- **v11 is running**: `sweep`, `--archs efficientnet_b0 mlp`, seeds 1 2 3, 224 px /
  60 epochs, timeout 43200 s. Started 05:15 UTC, expected to complete ~13:00 UTC.
  It mounts v10's output, so the split restores instead of rebuilding.
- Quota: 67,894 s used of 108,000 → **11.1 h left on this account**.
- Nothing local is required for it to finish; the session runs on Kaggle and its
  output (including `persist/`) is committed when the notebook ends.
- After it completes the GPU idles until someone pushes again. Do not push while
  v11 runs — poll `kernels/status` until `complete` first.
- Still missing: the second account (username + token at `~/.kaggle/team_token`).
  It must carry `resnet18_bilstm` (≈10 h) and `swin_tiny` (≈15-22 h, two
  sessions). This account cannot: after v11 it has ≈3 h left.

## Gotcha: pushing does NOT stop the running session

A push creates a *new batch session*; the old one keeps training and keeps
burning GPU quota. Two zombie sessions were found alive this way (v8, v9) and
the account hit `Maximum batch GPU session count of 2 reached`.

- `POST /api/i/kernels.KernelsService/CancelKernelSession` exists but the
  access token is refused: `Permission 'kernelSessions.cancel' was denied`.
  There is no cancel route in the public v1 API.
- So: never push while a session runs. Poll `kernels/status` until
  `complete`/`error`, then push. Cancelling has to happen in the Kaggle UI.
- A session killed by the 12 h cap does not save its `persist/`, so a stage that
  cannot finish inside ~11 h loses everything it trained.
- Size every session to ~11 h of work: one architecture's nine runs at 224/60
  is 8-10 h; Swin needs two sessions.

## Reading a run's log (Kaggle hides it until the session ends)

```bash
TOKEN=$(cat ~/.kaggle/access_token)
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://www.kaggle.com/api/v1/kernels/output?userName=utkarshkumarjha&kernelSlug=imgforge-metric-sweep" \
| python3 -c "
import json,sys,re
d=json.load(sys.stdin)
raw=(d.get('log') or '').replace('\n,',',').replace('\n',' ')
txt=''.join(e.get('data','') for e in json.loads(raw))
print(re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',txt)[-4000:])"
```

The output endpoint returns an empty log while a session is still running, and
403 for `urllib` without a `User-Agent` (curl is fine).

## Gotchas already paid for

- `save_notebook` (MCP) drops `datasetDataSources`; the REST `kernels/push`
  payload is the only way to attach CASIA and to mount the kernel's own output.
- The repo ships 65-row sample CSVs in `data/`, so the notebook only skips the
  split build when the CSVs came from `persist/`.
- The ELA cache is ~1.2 MB per image (~14 GB). It lives in `/tmp` via
  `--no-persist-ela-cache`; each session rebuilds it once (~15 min).
- `train.py` needs `--run-name` from the sweep, otherwise deviation-suffixed runs
  (pilot, reduced schedule) fail after training and overwrite the canonical
  checkpoint.
- Budget: quota refreshes 2026-10-31, i.e. **after** the submission deadline, so
  there is no second allotment to fall back on.

## Second account (teammate `mayurrdas`) — read this before pushing there

- The token is in the main checkout's `.env` as `MAYUR_TOKEN`, mirrored to
  `~/.kaggle/team_token` (chmod 600). Its owner is **`mayurrdas`** (double r);
  `mayurdas24` does not exist.
- The API derives a **new** kernel's slug from the `title` and ignores the
  owner prefix in `--slug`; on an existing kernel the slug must match exactly or
  the push is treated as a create and 409s on the duplicate title.
- `datasetDataSources` is **silently dropped** for kernels on this account
  (`invalidDatasetSources: []`, `SystemExit: CASIA2 not found`, v1-v3 all failed
  this way). The notebook now falls back to
  `kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset")`
  when `/kaggle/input` has no `CASIA2`, so the attach is optional. Paths stay
  stable because `prepare_dataset` runs against the
  `/kaggle/working/casia_link/CASIA2.0_revised` symlink, so the split
  fingerprint is unchanged.
- Self-source (`kernel_sources: [own kernel]`) is rejected with
  `invalidKernelSources` while the kernel has no completed version with output;
  it starts working once a run has saved `persist/`.
- This token cannot read `kernels/output` for another owner (`users.get`
  denied) — always query with the matching token.

## Session time budget

`run_sweep.py --max-hours H` stops *starting* new runs once H hours have
elapsed, after persisting the finished ones. Pass it through the notebook's
`ONLY` knob, e.g.
`--only "--archs resnet18_bilstm --max-hours 9.5"`. Without it a 12 h-cap kill
saves nothing.

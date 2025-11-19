# HouseGym RL

Reinforcement learning project for post-disaster housing recovery. The repo now follows a source/artifacts split so code stays clean while jobs drop all generated files under `artifacts/`.

## Layout

```
housegym_rl/
├── src/housegymrl/         # Active Python package (env, baselines, eval)
├── scripts/cluster/        # HiPerGator helpers (SLURM, upload/download, setup)
├── docs/guides/            # START_HERE + setup docs
├── experiments/            # Notebooks + ad-hoc analyses
├── data/                   # Input data (e.g., lombok_data.pkl)
└── artifacts/              # Models, checkpoints, results, logs, TensorBoard
```

Set `PYTHONPATH=src` (or run `pip install -e .` later) before running any script so `import housegymrl` works from everywhere.

## Fast Start

1. Install dependencies: `pip install -r requirements.txt`.
2. Export `PYTHONPATH=src` (add to your shell profile) so active modules resolve.
3. Explore configs in `src/housegymrl/config.py` and env logic in `src/housegymrl/housegymrl.py`.
4. Use `src/housegymrl/evaluate*.py` for local evaluation; they now save outputs into `artifacts/results` automatically.
5. Read `docs/guides/START_HERE.md` for the full workflow if you are ramping up.

## HiPerGator Workflow

1. **Sync code** – from repo root run `bash scripts/cluster/upload.sh`. It rsyncs `src/`, `scripts/`, `docs/`, and `experiments/` to `${HPG_BASE}` without touching `artifacts/` on the cluster.
2. **Setup environment** (first time per account) – login to HiPerGator and run:
   ```bash
   cd /home/yu.qianchen/ondemand/housegymrl/scripts/cluster
   bash setup_hpg.sh
   ```
   This script now validates the `src/housegymrl/` package, ensures `artifacts/*` folders exist, and installs the pinned dependencies in the `urbanai` conda env.
3. **Submit training** – still on the cluster run:
   ```bash
   cd /home/yu.qianchen/ondemand/housegymrl/scripts/cluster
   sbatch train_sac.slurm
   ```
   Logs stream into `/home/yu.qianchen/ondemand/housegymrl/artifacts/logs/train_<JOB_ID>.out`.
4. **Evaluate** – once checkpoints land in `artifacts/models/`, submit `sbatch evaluate_sac.slurm` to generate comparison CSVs/plots in `artifacts/results/`.
5. **Download artifacts** – back on local run `bash scripts/cluster/download_results.sh`. It mirrors `artifacts/{models,checkpoints,results,runs,logs}` from HiPerGator into the matching local directory.

## Key Scripts

- `scripts/cluster/train_sac_hpg.py` – main SAC trainer (now writing checkpoints to `artifacts/checkpoints/`, final models + VecNormalize to `artifacts/models/`, and the synthetic dataset to `artifacts/results/`).
- `scripts/cluster/train_sac.slurm` – SLURM wrapper that sets `PYTHONPATH=${BASE_DIR}/src` and runs the trainer inside the repo root.
- `scripts/cluster/evaluate_sac.slurm` – evaluation job that calls `src/housegymrl/evaluation.py` and saves everything under `artifacts/results/`.
- `scripts/cluster/setup_hpg.sh` – environment bootstrapper; it now checks the packaged layout instead of the old flat files.
- `scripts/cluster/upload.sh` / `download_results.sh` – rsync helpers aligned with the new tree.

## Artifacts

All generated assets now live at `artifacts/` to keep Git history clean:

- `artifacts/models/` – final policy zip + VecNormalize.
- `artifacts/checkpoints/` – interim SAC checkpoints.
- `artifacts/results/` – CSV summaries, curves, figs, RMSE dumps.
- `artifacts/runs/` – TensorBoard logs (`sac_diverse/` etc.).
- `artifacts/logs/` – SLURM stdout/stderr captured via job directives.

This makes it safe to `rsync` code without clobbering long-running experiments and keeps backups simple (just copy `artifacts`).

## Notes

- Anytime you add a new script under `src/housegymrl/`, remember to update `scripts/cluster/setup_hpg.sh` if additional files need to be validated on the cluster.
- When running locally, point any notebook or script output directories to `artifacts/...` so everything stays in one place.
- Historical documents now live under `docs/archive/` if you need to reference earlier reward/debug logs.

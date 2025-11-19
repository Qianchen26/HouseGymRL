# Reward/Observation Normalization

HouseGym RL now trains SAC with `VecNormalize` to stabilize learning across diverse regions.

## Training
- `scripts/cluster/train_sac_hpg.py` wraps `SubprocVecEnv` with `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=20, clip_reward=10, gamma=0.99)`.
- VecNormalize statistics are saved automatically to `artifacts/models/sac_diverse_vecnorm.pkl` alongside the trained policy zip.
- TensorBoard logs show normalized rewards (expect different scale vs raw completion).

## Evaluation
- Both `src/housegymrl/evaluate.py` (shim) and `evaluation.py` look for the saved stats path (defaults to `artifacts/models/sac_diverse_vecnorm.pkl`).
- During eval we load the stats via `VecNormalize.load(...)`, set `training=False`, and `norm_reward=False` so reported returns/metrics are in the true reward scale.
- If stats are missing, the scripts warn and fall back to an unnormalized env (results may deviate, so keep stats with the model).

## Workflow Tips
1. Always upload/download `sac_diverse_vecnorm.pkl` with the model zip (`upload.sh`/`download_results.sh` already do this).
2. When running ad-hoc local evals, ensure `PYTHONPATH=src` and that `artifacts/models/` contains both files.
3. If you fine-tune or retrain, delete the old stats file to avoid mixing normalization snapshots from different runs.

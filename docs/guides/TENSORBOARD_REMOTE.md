# Remote TensorBoard (HiPerGator)

Use the steps below whenever you want to inspect training logs stored in
`/home/yu.qianchen/ondemand/housegymrl/artifacts/runs` on HiPerGator.

## 1. Open SSH tunnel from local machine
```bash
ssh -L 1126:localhost:1126 yu.qianchen@hpg.rc.ufl.edu
```
- Replace the username if needed.
- Keep this terminal open; it forwards local port `1126` to the cluster.

## 2. Start TensorBoard on HiPerGator
In the same SSH session (after the prompt changes to `loginXX`):
```bash
module load conda
conda activate urbanai
tensorboard --logdir /home/yu.qianchen/ondemand/housegymrl/runs --port 1126

```
Leave TensorBoard running; it will report something like `http://localhost:1126/`.

## 3. View in browser
On your Mac, open a browser and navigate to:
```
http://localhost:1126
```
As long as the SSH tunnel + TensorBoard process stay alive, you will see live
training metrics.

## Notes
- If port 1126 is busy, pick another free port (e.g., 6007) and update both the
  `ssh -L` command and the `--port` flag accordingly.
- To stop, Ctrl+C TensorBoard first, then exit the SSH session.


到主层：cd /home/yu.qianchen/ondemand/housegymrl
监控：
tail -f artifacts/logs/train_<JOBID>.out
tail -f artifacts/logs/train_<JOBID>.err

tail -f artifacts/logs/train_18635201.out

查找是否开始训练：
rg -n "STARTING SAC TRAINING" artifacts/logs/train_<JOBID>.out
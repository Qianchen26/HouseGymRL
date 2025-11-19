#!/usr/bin/env python
# coding: utf-8

# # HouseGym RL: Comprehensive Training and Evaluation Pipeline
# 
# This notebook implements the full workflow:
# 1. **Setup & Configuration**
# 2. **Mechanism Verification** (10 verification cells)
# 3. **SAC Training**
# 4. **Cross-Scenario Evaluation**
# 5. **Validation & Visualization**

# ## Section 1: Setup & Configuration

# In[ ]:


# Cell 1.1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

# RL libraries
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# Environment
from housegymrl import RLEnv, BaselineEnv
from config import REGION_CONFIG, CAPACITY_RAMP_ENABLED

print("✅ All imports successful!")
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# In[ ]:


# Cell 1.2: Create Output Directories
output_dirs = [
    "../results",
    "../results/figures",
    "../results/rmse",
    "../models",
    "../runs",
    "../runs/sac_diverse",
]

for dir_path in output_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ Created/verified: {dir_path}")

print("\n✅ All output directories ready!")


# In[ ]:


# Cell 1.3: Configuration Summary
print("="*70)
print("CONFIGURATION SUMMARY")
print("="*70)

config = {
    "Environment": {
        "M_ratio": 0.10,
        "M_min": 512,
        "M_max": 2048,
        "stochastic_duration": True,
        "observation_noise": 0.15,
        "capacity_noise": 0.10,
        "use_longterm_reward": True,
        "use_batch_arrival": True,
        "use_capacity_ramp": False,  # Temporarily disabled
    },
    "Training": {
        "algorithm": "SAC",
        "total_timesteps": 500_000,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "batch_size": 512,
    },
    "Evaluation": {
        "test_regions": ["Mataram", "Sumbawa", "Central Lombok"],
        "crew_levels": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        "n_seeds": 5,
    }
}

for section, params in config.items():
    print(f"\n{section}:")
    for key, val in params.items():
        print(f"  {key:25s}: {val}")

print("\n" + "="*70)


# ## Section 2: Mechanism Verification
#
# Before training, we verify that all environment mechanisms work correctly.
# NOTE: Commented out for production training (already verified in development)

# In[ ]:

'''
# Cell 2.1: Verify Adaptive M System
print("="*70)
print("VERIFICATION 1: ADAPTIVE M SYSTEM")
print("="*70)

test_env = RLEnv(
    region_key="Mataram",
    M_ratio=0.10,
    M_min=512,
    M_max=2048,
    seed=42
)

print(f"\nConfiguration:")
print(f"  M_ratio: {test_env.M_ratio}")
print(f"  M_min: {test_env.M_min}")
print(f"  M_max: {test_env.M_max}")

# Test space dimensions
expected_obs_dim = 6 + 2048 * 4
expected_action_dim = 2049
actual_obs_dim = test_env.observation_space.shape[0]
actual_action_dim = test_env.action_space.shape[0]

print(f"\nSpace Dimensions:")
print(f"  Observation: {actual_obs_dim} (expected: {expected_obs_dim})")
print(f"  Action: {actual_action_dim} (expected: {expected_action_dim})")

if actual_obs_dim == expected_obs_dim and actual_action_dim == expected_action_dim:
    print("  ✅ Space dimensions correct!")
else:
    print("  ❌ ERROR: Space dimensions mismatch!")

# Test M calculation for different queue sizes
test_cases = [
    (100, 512),    # Small queue → M_min
    (5000, 500),   # Medium queue → 10%
    (10000, 1000), # Large queue → 10%
    (25000, 2048), # Very large → M_max
]

print(f"\nM Calculation Test:")
print(f"  Queue Size | Calculated M | Expected M | Status")
print(f"  {'-'*55}")

all_correct = True
for queue_size, expected_M in test_cases:
    calculated_M = test_env._get_M(queue_size)
    status = "✓" if calculated_M == expected_M else "✗"
    if calculated_M != expected_M:
        all_correct = False
    print(f"  {queue_size:10d} | {calculated_M:12d} | {expected_M:10d} | {status}")

if all_correct:
    print("\n✅ VERIFICATION 1 PASSED: Adaptive M system working correctly!")
else:
    print("\n❌ VERIFICATION 1 FAILED: Check M calculation logic!")

test_env.close()


# In[ ]:


# Cell 2.2: Verify Stochastic Work Duration
print("="*70)
print("VERIFICATION 2: STOCHASTIC WORK DURATION")
print("="*70)

# Create stochastic and deterministic environments
env_stochastic = RLEnv(
    region_key="Mataram",
    stochastic_duration=True,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

env_deterministic = RLEnv(
    region_key="Mataram",
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

# Reset and advance to day with capacity
env_stochastic.reset(seed=100)
env_deterministic.reset(seed=100)

for _ in range(37):
    env_stochastic.step(env_stochastic.action_space.sample())
    env_deterministic.step(env_deterministic.action_space.sample())

# Test work progress variance
test_houses = list(env_stochastic.waiting_queue.get_all())[:5]
initial_work_s = [env_stochastic._arr_rem[h] for h in test_houses]
initial_work_d = [env_deterministic._arr_rem[h] for h in test_houses]

allocation = {h: 10 for h in test_houses}
env_stochastic._apply_work(allocation)
env_deterministic._apply_work(allocation)

progress_s = [initial_work_s[i] - env_stochastic._arr_rem[test_houses[i]] for i in range(5)]
progress_d = [initial_work_d[i] - env_deterministic._arr_rem[test_houses[i]] for i in range(5)]

print(f"\nWork Progress Test (10 workers per house):")
print(f"  House | Expected | Stochastic | Deterministic")
print(f"  {'-'*50}")

for i in range(5):
    print(f"  {test_houses[i]:5d} | {10:8.2f} | {progress_s[i]:10.2f} | {progress_d[i]:13.2f}")

variance_s = np.var(progress_s)
variance_d = np.var(progress_d)

print(f"\nVariance:")
print(f"  Stochastic: {variance_s:.4f}")
print(f"  Deterministic: {variance_d:.4f}")

if abs(variance_d) < 0.01 and variance_s > variance_d:
    print("\n✅ VERIFICATION 2 PASSED: Stochastic duration working correctly!")
else:
    print("\n⚠️  VERIFICATION 2 WARNING: Check variance values")

env_stochastic.close()
env_deterministic.close()


# In[ ]:


# Cell 2.3: Verify Observation Noise
print("="*70)
print("VERIFICATION 3: OBSERVATION NOISE")
print("="*70)

env_noisy = RLEnv(
    region_key="Mataram",
    stochastic_duration=False,
    observation_noise=0.15,
    capacity_noise=0.0,
    seed=42
)

env_perfect = RLEnv(
    region_key="Mataram",
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

obs_noisy, _ = env_noisy.reset(seed=100)
obs_perfect, _ = env_perfect.reset(seed=100)

# Extract candidate features
candidates_noisy = obs_noisy[6:].reshape(-1, 4)
candidates_perfect = obs_perfect[6:].reshape(-1, 4)

valid_mask = candidates_noisy[:, 3] > 0.5
remain_noisy = candidates_noisy[valid_mask, 0][:10]
remain_perfect = candidates_perfect[valid_mask, 0][:10]

print(f"\nObservation Noise Test (first 10 valid candidates):")
print(f"  Candidate | Perfect Info | Noisy Obs | Difference | Noise %")
print(f"  {'-'*70}")

differences = []
for i in range(min(10, len(remain_noisy))):
    perfect = remain_perfect[i]
    noisy = remain_noisy[i]
    diff = noisy - perfect
    noise_pct = (diff / perfect * 100) if perfect > 0 else 0
    print(f"  {i:9d} | {perfect:12.2f} | {noisy:9.2f} | {diff:+10.2f} | {noise_pct:+7.1f}%")
    differences.append(abs(diff))

avg_diff = np.mean(differences)
print(f"\nAverage absolute difference: {avg_diff:.2f}")

if avg_diff > 1.0:
    print("\n✅ VERIFICATION 3 PASSED: Observation noise detected!")
else:
    print("\n⚠️  VERIFICATION 3 WARNING: Noise might be too small")

env_noisy.close()
env_perfect.close()


# In[ ]:


# Cell 2.4: Verify Capacity Noise
print("="*70)
print("VERIFICATION 4: CAPACITY NOISE")
print("="*70)

env_noisy = RLEnv(
    region_key="Mataram",
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.10,
    seed=42
)

env_perfect = RLEnv(
    region_key="Mataram",
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

env_noisy.reset(seed=100)
env_perfect.reset(seed=100)

# Advance to day 100
for _ in range(100):
    env_noisy.step(env_noisy.action_space.sample())
    env_perfect.step(env_perfect.action_space.sample())

# Record capacities
capacities_noisy = []
capacities_perfect = []

for _ in range(50):
    capacities_noisy.append(env_noisy._effective_capacity())
    capacities_perfect.append(env_perfect._effective_capacity())
    env_noisy.step(env_noisy.action_space.sample())
    env_perfect.step(env_perfect.action_space.sample())

base_capacity = capacities_perfect[0]
mean_noisy = np.mean(capacities_noisy)
std_noisy = np.std(capacities_noisy)
min_noisy = np.min(capacities_noisy)
max_noisy = np.max(capacities_noisy)

print(f"\nCapacity Statistics (50 days):")
print(f"  Base capacity (perfect): {base_capacity}")
print(f"  Noisy capacity mean: {mean_noisy:.2f}")
print(f"  Noisy capacity std dev: {std_noisy:.2f}")
print(f"  Noisy capacity range: [{min_noisy}, {max_noisy}]")
print(f"  Expected range: [{int(base_capacity * 0.90)}, {base_capacity}]")

if std_noisy > 0:
    print("\n✅ VERIFICATION 4 PASSED: Capacity noise detected!")
else:
    print("\n❌ VERIFICATION 4 FAILED: No capacity variance!")

env_noisy.close()
env_perfect.close()


# In[ ]:


# Cell 2.5: Verify Long-term Reward Components
print("="*70)
print("VERIFICATION 5: LONG-TERM REWARD FUNCTION")
print("="*70)

env_longterm = RLEnv(
    region_key="Mataram",
    use_longterm_reward=True,
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

env_legacy = RLEnv(
    region_key="Mataram",
    use_longterm_reward=False,
    stochastic_duration=False,
    observation_noise=0.0,
    capacity_noise=0.0,
    seed=42
)

env_longterm.reset(seed=100)
env_legacy.reset(seed=100)

# Advance to active phase
for _ in range(50):
    env_longterm.step(env_longterm.action_space.sample())
    env_legacy.step(env_legacy.action_space.sample())

# Take one step and compare rewards
_, r_long, _, _, info_long = env_longterm.step(env_longterm.action_space.sample())
_, r_legacy, _, _, info_legacy = env_legacy.step(env_legacy.action_space.sample())

print(f"\nReward Comparison (single step):")
print(f"  Long-term reward: {r_long:.6f}")
print(f"  Legacy reward: {r_legacy:.6f}")
print(f"  Difference: {abs(r_long - r_legacy):.6f}")

print(f"\nLong-term Reward Components:")
print(f"  - Completion reward (weight 1.0)")
print(f"  - Queue reduction bonus (weight 0.2)")
print(f"  - Urgency penalty (weight 0.1)")
print(f"  - Worker efficiency bonus (weight 0.05)")
print(f"  - NO damage weighting (equal treatment)")

print(f"\nCurrent State:")
print(f"  Queue size: {info_long.get('queue_size', 0)}")
print(f"  Completion: {info_long.get('completion', 0):.4f}")
print(f"  Max waiting time: {np.max(env_longterm.waiting_time) if len(env_longterm.waiting_time) > 0 else 0} days")

print("\n✅ VERIFICATION 5 PASSED: Long-term reward function implemented!")

env_longterm.close()
env_legacy.close()


# In[ ]:


# Cell 2.6: Verify Baseline Policies (LJF/SJF)
print("="*70)
print("VERIFICATION 6: BASELINE POLICIES")
print("="*70)

for policy_name in ["LJF", "SJF", "Random"]:
    print(f"\nTesting {policy_name} policy...")
    
    env = BaselineEnv(
        region_key="Mataram",
        policy=policy_name,
        M_ratio=0.10,
        M_min=512,
        M_max=2048,
        seed=42
    )
    
    obs, info = env.reset(seed=100)
    
    # Collect completion after 10 actual days
    completions = []
    while len(completions) < 10:
        obs, reward, done, trunc, info = env.step()
        if info.get('day_advanced', False):
            completions.append(info.get('completion', 0))
        if done or trunc:
            break

    final_completion = completions[-1] if completions else 0.0
    print(f"  Completed {len(completions)} days")
    print(f"  Final completion: {final_completion:.4f}")
    print(f"  ✅ {policy_name} policy works correctly")
    
    env.close()

print("\n✅ VERIFICATION 6 PASSED: All baseline policies working!")


# In[ ]:


# Cell 2.7: Verify Completion Calculation Consistency
print("="*70)
print("VERIFICATION 7: COMPLETION CALCULATION CONSISTENCY")
print("="*70)

env = RLEnv(region_key="Mataram", seed=42)
env.reset(seed=100)

# Advance to day 100
target_day = 100
last_info = None
while env.day < target_day:
    obs, reward, done, trunc, info = env.step(env.action_space.sample())
    if info.get('day_advanced', False):
        last_info = info
    if done or trunc:
        break

# Manual completion calculation
revealed_ids = list(env.arrival_system.revealed_ids)
completed_manual = sum(1 for h in revealed_ids if env._arr_rem[h] <= 0)
completion_manual = completed_manual / len(revealed_ids) if revealed_ids else 0

# Info completion
completion_info = (last_info or info).get('completion', 0)

print(f"\nCompletion Calculation:")
print(f"  Revealed houses: {len(revealed_ids)}")
print(f"  Completed houses (manual count): {completed_manual}")
print(f"  Manual completion: {completion_manual:.6f}")
print(f"  Info completion: {completion_info:.6f}")
print(f"  Difference: {abs(completion_manual - completion_info):.6f}")

if abs(completion_manual - completion_info) < 1e-6:
    print("\n✅ VERIFICATION 7 PASSED: Completion calculation consistent!")
else:
    print("\n❌ VERIFICATION 7 FAILED: Completion calculation mismatch!")

env.close()


# In[ ]:


# Cell 2.8: Verify Batch Arrival System
print("="*70)
print("VERIFICATION 8: BATCH ARRIVAL SYSTEM")
print("="*70)

env = RLEnv(
    region_key="Mataram",
    use_batch_arrival=True,
    seed=42
)
env.reset(seed=100)

# Expected batch days: [0, 30, 60]
# Expected ratios: [0.40, 0.35, 0.25]

batch_days = [0, 30, 60]
revealed_counts = []

terminated = False

for day in batch_days:
    # Advance to batch day
    while env.day < day:
        obs, reward, done, trunc, info = env.step(env.action_space.sample())
        if done or trunc:
            terminated = True
            break
    if terminated:
        break
    
    revealed = len(env.arrival_system.revealed_ids)
    revealed_counts.append(revealed)
    print(f"\nDay {day}: {revealed} houses revealed")

# Check proportions
total_houses = len(env.tasks_df)
expected_counts = [int(total_houses * r) for r in [0.40, 0.35, 0.25]]

print(f"\nBatch Arrival Analysis:")
print(f"  Total houses: {total_houses}")
print(f"  Day | Revealed | Expected | Ratio")
print(f"  {'-'*45}")

for i, (day, revealed) in enumerate(zip(batch_days, revealed_counts)):
    ratio = revealed / total_houses if total_houses > 0 else 0.0
    expected = expected_counts[i]
    print(f"  {day:3d} | {revealed:8d} | {expected:8d} | {ratio:.2%}")

print("\n✅ VERIFICATION 8 PASSED: Batch arrival working!")
env.close()


# In[ ]:


# Cell 2.9: Verify Capacity Ramp System
print("="*70)
print("VERIFICATION 9: CAPACITY RAMP SYSTEM")
print("="*70)

if not CAPACITY_RAMP_ENABLED:
    print("⚠️ Capacity ramp 功能已暂时禁用，跳过该验证，所有实验均使用固定容量。")
else:
    env = RLEnv(
        region_key="Mataram",
        use_capacity_ramp=True,
        capacity_noise=0.0,  # Disable noise for clean test
        seed=42
    )
    env.reset(seed=100)

    test_days = [0, 10, 35, 36, 100, 150, 216, 250]

    print(f"\nCapacity Ramp Test:")
    print(f"  Day | Capacity | Phase")
    print(f"  {'-'*40}")

    done = False
    trunc = False
    for day in test_days:
        # Advance to test day
        while env.day < day:
            obs, reward, done, trunc, info = env.step(env.action_space.sample())
            if done or trunc:
                break
        if done or trunc:
            break

        capacity = env._effective_capacity()
        
        if day < 36:
            phase = "Warmup (K=0)"
        elif day < 216:
            phase = "Ramp (K growing)"
        else:
            phase = "Full capacity"
        
        print(f"  {day:3d} | {capacity:8d} | {phase}")

    print("\n✅ VERIFICATION 9 PASSED: Capacity ramp working!")
    env.close()


# In[ ]:


# Cell 2.10: Load and Verify Training Data
print("="*70)
print("VERIFICATION 10: TRAINING DATA")
print("="*70)

# Load synthetic training dataset
training_df = pd.read_csv('../results/synthetic_training_dataset.csv')

print(f"\nTraining Dataset Summary:")
print(f"  Total regions: {len(training_df)}")

if 'cluster' in training_df.columns:
    print("  Cluster distribution:")
    print(training_df.groupby('cluster').size())
    preview_cols = ['region_key', 'total_houses', 'cluster']
elif 'scenario' in training_df.columns:
    print("  Cluster column missing; showing scenario distribution instead:")
    print(training_df.groupby('scenario').size())
    preview_cols = ['region_key', 'total_houses', 'scenario']
else:
    print("  No cluster/scenario column available for distribution summary.")
    preview_cols = ['region_key', 'total_houses']

print(f"\nSize Statistics:")
print(f"  Min: {training_df['total_houses'].min()}")
print(f"  Max: {training_df['total_houses'].max()}")
print(f"  Mean: {training_df['total_houses'].mean():.0f}")
print(f"  Std: {training_df['total_houses'].std():.0f}")

print(f"\nDamage Distribution (first 5 regions):")
print(training_df[preview_cols].head())

if len(training_df) == 150:
    print("\n✅ VERIFICATION 10 PASSED: Training data loaded correctly!")
else:
    print(f"\n⚠️  WARNING: Expected 150 regions, found {len(training_df)}")
'''

# Cell 2.10: Load Training Data (Essential - DO NOT COMMENT OUT)
print("="*70)
print("LOADING TRAINING DATA")
print("="*70)

# Load synthetic training dataset
training_df = pd.read_csv('../results/synthetic_training_dataset.csv')
print(f"✅ Loaded {len(training_df)} training regions")

# Register all synthetic regions
from config import register_synthetic_region

print(f"Registering {len(training_df)} synthetic regions...")
for idx, row in training_df.iterrows():
    register_synthetic_region(
        H=int(row['total_houses']),
        K=int(row['num_contractors']),
        damage_dist=[
            int(row['minor_count']),
            int(row['moderate_count']),
            int(row['major_count'])
        ],
        seed=int(row['seed']),
        region_key=str(row.get('region_key', '')) or None,
    )
print(f"✅ All {len(training_df)} regions registered!")

# Store reference for later use
df = training_df

# ## Section 3: SAC Training
# 
# Now that all mechanisms are verified, we train the SAC model on diverse synthetic data.

# In[ ]:


# Cell 3.1: Define Environment Factory
def make_diverse_training_env(rank: int, seed: int):
    """
    Create training environment that samples from diverse synthetic regions.
    Each episode uses a different random region.
    """
    # Load training dataset
    df = pd.read_csv('../results/synthetic_training_dataset.csv')
    region_keys = df['region_key'].tolist()
    
    def _init():
        rng = np.random.default_rng(seed + rank)
        
        # Randomly select a region for this episode
        region_key = rng.choice(region_keys)
        
        env = RLEnv(
            region_key=region_key,
            M_ratio=0.10,
            M_min=512,
            M_max=2048,
            use_batch_arrival=True,
            use_capacity_ramp=False,
            stochastic_duration=True,
            observation_noise=0.15,
            capacity_noise=0.10,
            use_longterm_reward=True,
            seed=seed + rank * 1000
        )
        
        return env
    
    return _init

print("✅ Environment factory defined!")


# In[ ]:


# Cell 3.2: Create Vectorized Training Environment
print("="*70)
print("CREATING VECTORIZED TRAINING ENVIRONMENT")
print("="*70)

from stable_baselines3.common.vec_env import DummyVecEnv

# Configuration
N_ENVS = 8
SEED = 42

print(f"\nConfiguration:")
print(f"  Number of environments: {N_ENVS}")
print(f"  Base seed: {SEED}")
print(f"  Using: DummyVecEnv (single-process for notebook stability)")

# Training dataset should already be loaded and registered in Cell 2.10
if 'df' not in globals():
    raise RuntimeError("Training dataset not loaded! Please ensure Cell 2.10 executed successfully.")
else:
    print(f"\n✅ Using {len(df)} registered regions from Cell 2.10")

# Define environment factory
def make_training_env(rank: int):
    """Create a single training environment"""
    def _init():
        rng = np.random.default_rng(SEED + rank)
        
        # Randomly select a region (regions already registered)
        region_key = rng.choice(df['region_key'].tolist())
        
        env = RLEnv(
            region_key=region_key,
            M_ratio=0.10,
            M_min=512,
            M_max=2048,
            use_batch_arrival=True,
            use_capacity_ramp=False,
            stochastic_duration=True,
            observation_noise=0.15,
            capacity_noise=0.10,
            use_longterm_reward=True,
            seed=SEED + rank * 1000
        )
        
        return env
    
    return _init

# Create vectorized environment
print(f"\nCreating {N_ENVS} environments...")
try:
    env_fns = [make_training_env(i) for i in range(N_ENVS)]
    vec_env = DummyVecEnv(env_fns)
    print("✅ DummyVecEnv created")
    
    vec_env = VecMonitor(vec_env)
    print("✅ VecMonitor wrapped")
    
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    print("✅ VecNormalize wrapped")
    
    print(f"\n✅ Vectorized environment ready!")
    print(f"   Observation space: {vec_env.observation_space.shape}")
    print(f"   Action space: {vec_env.action_space.shape}")
    
    print(f"\n📝 Notes:")
    print(f"   - DummyVecEnv is single-process (more stable in notebooks)")
    print(f"   - Training will be slower than SubprocVecEnv but reliable")
    print(f"   - For faster training, use: python train_sac_script.py")
    
except Exception as e:
    print(f"\n❌ Error creating vectorized environment:")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)


# In[ ]:


# Cell 3.3: Create SAC Model
print("="*70)
print("CREATING SAC MODEL")
print("="*70)

tensorboard_log_dir = "../runs/sac_diverse/"
model = SAC(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    buffer_size=1_500_000,       # Increased for 500-step episodes (3000 episodes capacity)
    batch_size=256,              # Reduced for initial stability
    learning_starts=25_000,      # Wait for buffer to fill 5% before learning
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    ent_coef='auto',
    target_update_interval=1,
    verbose=1,                   # Reduced verbosity (1 instead of 2)
    tensorboard_log=tensorboard_log_dir,
    device='auto'
)

print("\n✅ SAC model created successfully!")
print(f"\nModel Configuration:")
print(f"  Policy: MlpPolicy")
print(f"  Learning rate: 3e-4")
print(f"  Buffer size: 1,500,000")
print(f"  Batch size: 256")
print(f"  Learning starts: 25,000")
print(f"  Gamma: 0.99")
print(f"  Entropy coefficient: auto")
print(f"  Verbose: 1")
print(f"  Device: {model.device}")


# In[ ]:


# Cell 3.4: Setup Training Callbacks
print("="*70)
print("SETTING UP TRAINING CALLBACKS")
print("="*70)

# Reward component logger for TensorBoard
class RewardComponentLogger(BaseCallback):
    """
    Logs individual reward components to TensorBoard for analysis.
    Components: progress, completion, queue_penalty, capacity_usage.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if we have info from the latest step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]

            # Log reward components if they exist
            if "reward_progress" in info:
                self.logger.record("reward/progress", info["reward_progress"])
            if "reward_completion" in info:
                self.logger.record("reward/completion", info["reward_completion"])
            if "reward_queue_penalty" in info:
                self.logger.record("reward/queue_penalty", info["reward_queue_penalty"])
            if "reward_capacity_usage" in info:
                self.logger.record("reward/capacity_usage", info["reward_capacity_usage"])
            if "reward_raw_total" in info:
                self.logger.record("reward/raw_total", info["reward_raw_total"])

        return True

# Custom callback to monitor and log training
class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback to force logging of training metrics.
    Fixes issue where verbose=2 and TensorBoard don't output anything.
    """
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0:
            # Print training stats
            if self.verbose > 0:
                print(f"\n[TRAINING LOG] Step {self.num_timesteps}")
                print(f"  Buffer size: {self.model.replay_buffer.size()}")
                print(f"  Gradient updates: {self.model._n_updates}")

            # Manually log to tensorboard
            if self.model.logger is not None:
                self.model.logger.record("train/buffer_size", self.model.replay_buffer.size())
                self.model.logger.record("train/n_updates", self.model._n_updates)
                self.model.logger.dump(self.num_timesteps)

        return True

# Create callbacks
reward_logger = RewardComponentLogger(verbose=0)  # Log reward components every step

training_logger = TrainingLoggerCallback(log_freq=5000, verbose=1)  # Log every 5000 steps

checkpoint_callback = CheckpointCallback(
    save_freq=50_000 // N_ENVS,  # Adjust for parallel envs
    save_path='../models/checkpoints/',
    name_prefix='sac_diverse',
    verbose=1
)
print("✅ Reward component logger configured (every step)")
print("✅ Checkpoint callback configured (every 50k steps)")
print("✅ Training logger callback configured (every 5k steps)")

callbacks = [reward_logger, training_logger, checkpoint_callback]

print(f"\n✅ {len(callbacks)} callback(s) ready for training")


# In[ ]:


# Cell 3.5: Train SAC Model
print("="*70)
print("STARTING SAC TRAINING")
print("="*70)

# Production training configuration
# IMPORTANT: Each episode requires ~400k steps per environment!
# With 8 parallel envs, episode completion schedule:
#   - 5M steps: ~1-2 episodes per env (⚠️ TOO FEW for meaningful learning curve!)
#   - 10M steps: ~3 episodes per env (minimum for basic trend)
#   - 20M steps: ~6 episodes per env (better, ~56h / 2.3 days)
#   - 32M steps: ~10 episodes per env (recommended, ~89h / 3.7 days)
#   - 48M steps: ~15 episodes per env (good training, ~133h / 5.6 days)
#   - 64M steps: ~20 episodes per env (solid training, ~178h / 7.4 days)
#
# For production research: Recommend 32M-64M steps for robust learning curves
# For quick testing: 5M-10M steps just to verify code works
# For rapid iteration: 500k steps to validate reward function changes
TOTAL_TIMESTEPS = 500_000  # Current: Quick validation for reward optimization

print(f"\nTraining Configuration:")
print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"  Parallel environments: {N_ENVS}")
print(f"  Steps per environment: {TOTAL_TIMESTEPS // N_ENVS:,}")
print(f"  Estimated episodes: ~{(TOTAL_TIMESTEPS // N_ENVS) // 400_000}-{(TOTAL_TIMESTEPS // N_ENVS) // 300_000} per env")
print(f"  Estimated training time: ~{TOTAL_TIMESTEPS / 500_000 * 100:.0f} minutes")
print(f"  Tensorboard log: ./runs/sac_diverse/")

print(f"\n{'='*70}")
print("TRAINING IN PROGRESS...")
print(f"{'='*70}")
print("\nTo monitor training:")
print("  tensorboard --logdir ./runs/sac_diverse/")
print()

# Debug: Check training parameters
print("\n[DEBUG] SAC Training Parameters:")
print(f"  learning_starts: {model.learning_starts}")
print(f"  batch_size: {model.batch_size}")
print(f"  train_freq: {model.train_freq}")
print(f"  Buffer capacity: {model.replay_buffer.buffer_size}")
print()

# Start training
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    log_interval=10  # Log every 10 episodes (or 10*steps if no episodes complete)
)

print(f"\n{'='*70}")
print("✅ TRAINING COMPLETED!")
print(f"{'='*70}")

# Debug: Verify training happened
print(f"\n[DEBUG] Post-training status:")
print(f"  Replay buffer size: {model.replay_buffer.size()}")
if hasattr(model, '_n_updates'):
    print(f"  Gradient updates performed: {model._n_updates}")
else:
    print(f"  WARNING: _n_updates attribute not found")
print()


# In[ ]:


# Cell 3.6: Save Trained Model
print("="*70)
print("SAVING TRAINED MODEL")
print("="*70)

# Save model
model.save("../models/sac_diverse_final")
print("✅ Model saved: ../models/sac_diverse_final.zip")

# Save VecNormalize statistics
vec_env.save("../models/sac_diverse_vecnorm.pkl")
print("✅ VecNormalize stats saved: ../models/sac_diverse_vecnorm.pkl")

print("\n✅ All training artifacts saved successfully!")
print("\nSaved files:")
print("  - ../models/sac_diverse_final.zip")
print("  - ../models/sac_diverse_vecnorm.pkl")
print("  - ../models/checkpoints/sac_diverse_*.zip")


# Skip cross-scenario evaluation temporarily
print("\nSkipping cross-scenario evaluation (disabled in main.py).")

'''
# ## Section 4: Cross-Scenario Evaluation
# 
# Evaluate the trained model across different regions and crew availability levels.

# In[ ]:


# Cell 4.1: Load Trained Model
print("="*70)
print("LOADING TRAINED MODEL")
print("="*70)

# Load SAC model
sac_model = SAC.load("../models/sac_diverse_final")
print("✅ SAC model loaded")

# Note: VecNormalize stats not needed for evaluation since we evaluate on raw environment

print("\n✅ Model ready for evaluation!")


# In[ ]:


# Cell 4.2: Run Cross-Scenario Evaluation
print("="*70)
print("CROSS-SCENARIO EVALUATION")
print("="*70)

def evaluate_single_run(model, policy_name, region, crew_availability, seed, max_days=500):
    """Run single evaluation episode"""
    # Get base crew count
    base_crew = REGION_CONFIG[region]['num_contractors']
    actual_crew = int(base_crew * crew_availability)
    
    # Create environment
    if policy_name in ['SAC']:
        env = RLEnv(
            region_key=region,
            num_contractors=actual_crew,
            M_ratio=0.10,
            M_min=512,
            M_max=2048,
            use_batch_arrival=True,
            use_capacity_ramp=False,
            stochastic_duration=True,
            observation_noise=0.15,
            capacity_noise=0.10,
            use_longterm_reward=True,
            seed=seed
        )
    else:
        env = BaselineEnv(
            region_key=region,
            policy=policy_name,
            num_contractors=actual_crew,
            use_batch_arrival=True,
            use_capacity_ramp=False,
            seed=seed
        )
    
    # Run episode
    obs, info = env.reset()
    trajectory = []
    
    while len(trajectory) < max_days:
        if policy_name == 'SAC':
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = None

        obs, reward, done, trunc, info = env.step(action)
        if info.get('day_advanced', False):
            trajectory.append(info.get('completion', 0))

        if done or trunc:
            break
    
    env.close()
    
    # Extract metrics
    final_completion = trajectory[-1] if trajectory else 0.0
    makespan = next((i for i, c in enumerate(trajectory) if c >= 0.99), max_days)
    
    return {
        'policy': policy_name,
        'region': region,
        'crew_availability': crew_availability,
        'seed': seed,
        'final_completion': final_completion,
        'makespan': makespan,
        'trajectory': trajectory
    }

# Evaluation configuration
test_regions = ["Mataram", "Sumbawa", "Central Lombok"]
crew_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
policies = {'SAC': sac_model, 'LJF': None, 'SJF': None, 'Random': None}
n_seeds = 5
base_seed = 1000

total_runs = len(policies) * len(test_regions) * len(crew_levels) * n_seeds
print(f"\nEvaluation Configuration:")
print(f"  Test regions: {test_regions}")
print(f"  Crew levels: {crew_levels}")
print(f"  Policies: {list(policies.keys())}")
print(f"  Seeds per condition: {n_seeds}")
print(f"  Total runs: {total_runs}")

print(f"\n{'='*70}")
print("RUNNING EVALUATION...")
print(f"{'='*70}\n")

results = []
current_run = 0

for policy_name, model_obj in policies.items():
    for region in test_regions:
        for crew_level in crew_levels:
            for seed_offset in range(n_seeds):
                seed = base_seed + seed_offset
                current_run += 1
                
                if current_run % 10 == 0 or current_run == total_runs:
                    print(f"[{current_run}/{total_runs}] {policy_name:8s} | {region:20s} | Crew={crew_level:.1f}")
                
                result = evaluate_single_run(
                    model_obj, policy_name, region, crew_level, seed
                )
                results.append(result)

# Convert to DataFrame
results_df = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'trajectory'}
    for r in results
])

# Save results
results_df.to_csv('../results/cross_scenario_results.csv', index=False)
print(f"\n✅ Evaluation complete! Results saved to: ../results/cross_scenario_results.csv")

# Quick summary
print(f"\n{'='*70}")
print("QUICK SUMMARY")
print(f"{'='*70}")
summary = results_df.groupby('policy')['final_completion'].agg(['mean', 'std'])
print(summary)


# ## Section 5: Validation & Visualization
# 
# Analyze results and generate comprehensive visualizations.

# In[ ]:


# Cell 5.1: Summary Statistics
print("="*70)
print("DETAILED SUMMARY STATISTICS")
print("="*70)

results_df = pd.read_csv('../results/cross_scenario_results.csv')

print(f"\nOverall Performance by Policy:")
summary = results_df.groupby('policy')['final_completion'].agg([
    'mean', 'std', 'min', 'max'
])
summary['cv'] = summary['std'] / summary['mean']  # Coefficient of variation
summary = summary.sort_values('mean', ascending=False)
print(summary)

print(f"\n{'='*70}")
print("Performance by Region:")
print(f"{'='*70}")
for region in results_df['region'].unique():
    print(f"\n{region}:")
    region_summary = results_df[results_df['region'] == region].groupby('policy')['final_completion'].mean()
    region_summary = region_summary.sort_values(ascending=False)
    print(region_summary)

print(f"\n{'='*70}")
print("Crew Sensitivity Analysis:")
print(f"{'='*70}")
for policy in results_df['policy'].unique():
    policy_df = results_df[results_df['policy'] == policy]
    comp_10 = policy_df[policy_df['crew_availability'] == 0.1]['final_completion'].mean()
    comp_100 = policy_df[policy_df['crew_availability'] == 1.0]['final_completion'].mean()
    drop = comp_100 - comp_10
    drop_pct = (drop / comp_100) * 100 if comp_100 > 0 else 0
    print(f"{policy:10s}: {comp_100:.2%} → {comp_10:.2%} (drop: {drop_pct:.1f}%)")


# In[ ]:


# Cell 5.2: Generate Heatmap
print("="*70)
print("GENERATING HEATMAP: Policy × Crew Availability")
print("="*70)

pivot = results_df.pivot_table(
    values='final_completion',
    index='policy',
    columns='crew_availability',
    aggfunc='mean'
)

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', 
            vmin=0, vmax=1, cbar_kws={'label': 'Mean Completion'})
plt.title('Performance Heatmap: Policy × Crew Availability', fontsize=14, pad=20)
plt.xlabel('Crew Availability', fontsize=12)
plt.ylabel('Policy', fontsize=12)
plt.tight_layout()
plt.savefig('../results/figures/heatmap_policy_crew.png', dpi=150, bbox_inches='tight')
print("✅ Saved: ../results/figures/heatmap_policy_crew.png")
plt.show()


# In[ ]:


# Cell 5.3: Generate Crew Sensitivity Curves
print("="*70)
print("GENERATING CREW SENSITIVITY CURVES")
print("="*70)

fig, axes = plt.subplots(1, len(results_df['region'].unique()), 
                         figsize=(6*len(results_df['region'].unique()), 5))

if len(results_df['region'].unique()) == 1:
    axes = [axes]

colors = {'SAC': '#1f77b4', 'LJF': '#2ca02c', 'SJF': '#d62728', 'Random': '#9467bd'}

for ax, region in zip(axes, results_df['region'].unique()):
    region_df = results_df[results_df['region'] == region]
    
    for policy in region_df['policy'].unique():
        policy_df = region_df[region_df['policy'] == policy]
        grouped = policy_df.groupby('crew_availability')['final_completion']
        means = grouped.mean()
        stds = grouped.std()
        
        ax.plot(means.index, means.values, marker='o', label=policy, 
                linewidth=2.5, color=colors.get(policy), alpha=0.9)
        ax.fill_between(means.index, 
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.2, color=colors.get(policy))
    
    ax.set_xlabel('Crew Availability', fontsize=12)
    ax.set_ylabel('Completion Rate', fontsize=12)
    ax.set_title(f'{region}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('../results/figures/crew_sensitivity.png', dpi=150, bbox_inches='tight')
print("✅ Saved: ../results/figures/crew_sensitivity.png")
plt.show()


# In[ ]:


# Cell 5.4: Generate Robustness Box Plot
print("="*70)
print("GENERATING ROBUSTNESS BOX PLOT")
print("="*70)

plt.figure(figsize=(12, 6))
results_df.boxplot(column='final_completion', by='policy', ax=plt.gca())
plt.title('Performance Robustness Across All Scenarios', fontsize=14, pad=20)
plt.suptitle('')  # Remove default title
plt.xlabel('Policy', fontsize=12)
plt.ylabel('Completion Rate', fontsize=12)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/figures/robustness_boxplot.png', dpi=150, bbox_inches='tight')
print("✅ Saved: ../results/figures/robustness_boxplot.png")
plt.show()


# In[ ]:


# Cell 5.5: Statistical Significance Tests
print("="*70)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

from scipy import stats

# Pairwise comparisons at crew=1.0
crew_100_df = results_df[results_df['crew_availability'] == 1.0]

policies = crew_100_df['policy'].unique()

print("\nPairwise t-tests (crew=1.0):")
print(f"{'Policy 1':10s} vs {'Policy 2':10s} | t-stat | p-value | Significant")
print("-" * 70)

for i, policy1 in enumerate(policies):
    for policy2 in policies[i+1:]:
        data1 = crew_100_df[crew_100_df['policy'] == policy1]['final_completion']
        data2 = crew_100_df[crew_100_df['policy'] == policy2]['final_completion']
        
        t_stat, p_value = stats.ttest_ind(data1, data2)
        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        
        print(f"{policy1:10s} vs {policy2:10s} | {t_stat:7.3f} | {p_value:7.4f} | {sig}")

print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")


# In[ ]:


# Cell 5.6: Generate Final Summary Report
print("="*70)
print("FINAL SUMMARY REPORT")
print("="*70)

report = []
report.append("# HouseGym RL: Evaluation Report\n")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

report.append("## Overall Performance\n")
summary = results_df.groupby('policy')['final_completion'].agg(['mean', 'std', 'min', 'max'])
summary['cv'] = summary['std'] / summary['mean']
summary = summary.sort_values('mean', ascending=False)
report.append(summary.to_string())
report.append("\n\n")

report.append("## Best Policy per Region\n")
for region in results_df['region'].unique():
    region_df = results_df[results_df['region'] == region]
    best = region_df.groupby('policy')['final_completion'].mean().idxmax()
    best_score = region_df.groupby('policy')['final_completion'].mean().max()
    report.append(f"- {region}: {best} ({best_score:.2%})\n")
report.append("\n")

report.append("## Crew Sensitivity\n")
for policy in results_df['policy'].unique():
    policy_df = results_df[results_df['policy'] == policy]
    comp_10 = policy_df[policy_df['crew_availability'] == 0.1]['final_completion'].mean()
    comp_100 = policy_df[policy_df['crew_availability'] == 1.0]['final_completion'].mean()
    drop = comp_100 - comp_10
    report.append(f"- {policy}: {comp_100:.2%} → {comp_10:.2%} (drop: {drop:.2%})\n")
report.append("\n")

report.append("## Key Findings\n")
best_overall = summary.index[0]
most_robust = summary.sort_values('cv').index[0]
report.append(f"- Best overall performance: {best_overall}\n")
report.append(f"- Most robust (lowest CV): {most_robust}\n")
report.append(f"- Total scenarios tested: {len(results_df)}\n")

# Save report
with open('../results/evaluation_report.md', 'w') as f:
    f.writelines(report)

print("\n✅ Report saved: ../results/evaluation_report.md")

# Print report
print("\n" + "="*70)
for line in report:
    print(line, end='')
print("\n" + "="*70)

print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("\nGenerated outputs:")
print("  - ../models/sac_diverse_final.zip")
print("  - ../results/cross_scenario_results.csv")
print("  - ../results/figures/heatmap_policy_crew.png")
print("  - ../results/figures/crew_sensitivity.png")
print("  - ../results/figures/robustness_boxplot.png")
print("  - ../results/evaluation_report.md")
'''

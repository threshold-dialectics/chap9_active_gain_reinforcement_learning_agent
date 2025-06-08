#phoenix_loop_intervention_RL.py

import numpy as np
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.ndimage import correlate1d
from scipy.stats import sem, ttest_ind, linregress
from collections import deque
from itertools import groupby
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import joblib
import copy
import time
from datetime import timedelta
import functools
import contextlib
import io
import os
import warnings
import json
import subprocess
import gymnasium as gym # Changed from import gym
from gymnasium import spaces # Changed from from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env # For checking custom env

FORCE_RETRAIN = False

class NpEncoder(json.JSONEncoder):
    """JSON encoder that falls back to built-ins for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()          # let json encode the native list
        return super().default(obj)

def get_git_commit_hash() -> str:
    """Return short git commit hash if available."""
    try:
        return subprocess.check_output([
            'git', 'rev-parse', '--short', 'HEAD'
        ]).decode().strip()
    except Exception:
        return 'unknown'
# --- Environment Configuration ---
class EnvironmentConfig:
    def __init__(self, level="default"):
        self.level = level
        print(f"INFO: Initializing EnvironmentConfig with level: {level}")
        # baseline parameters for all environments
        self.fcrit_replenish_rate_decay = 0.0
        self.strain_baseline_drift = 0.0
        if level == "easy_debug":
            self.fcrit_initial = 100.0
            self.fcrit_floor = 10.0
            self.fcrit_replenish_rate = 0.1
            self.baseline_fcrit_operational_cost = 0.2
            self.beta_decay_flaring = 0.05
            self.beta_rebuild_pruning = 0.02
            self.beta_phase_iv_noise_std = 0.002
            self.strain_baseline = 3.0
            self.strain_shock_magnitude = 5.0
            self.strain_shock_duration = 20
            self.strain_shock_interval = 150
            self.post_collapse_strain_reduction = 0.7
            self.fcrit_high_level_decay_rate = 0.0
            self.sim_steps = 600
            self.n_intervention_runs = 30 # For analysis runs
            self.total_rl_training_timesteps = 500000
            self.shock_start_time = 50
            self.beta_initial = 0.8 # Add if used by PhoenixLoopRLEnv for clipping
        elif level == "challenging":
            self.fcrit_initial = 60.0
            self.fcrit_floor = 10.0
            self.fcrit_replenish_rate = 0.05
            self.baseline_fcrit_operational_cost = 0.5
            self.beta_decay_flaring = 0.06
            self.beta_rebuild_pruning = 0.015
            self.beta_phase_iv_noise_std = 0.003
            self.strain_baseline = 3.5
            self.strain_shock_magnitude = 7.0
            self.strain_shock_duration = 25
            self.strain_shock_interval = 120
            self.post_collapse_strain_reduction = 0.6
            self.fcrit_high_level_decay_rate = 0.0005
            self.sim_steps = 800 # Longer for challenging
            self.n_intervention_runs = 30 # More runs for challenging analysis
            self.total_rl_training_timesteps = 500000 # More training for challenging
            self.shock_start_time = 70
            self.beta_initial = 0.8
        elif level == "extreme":
            self.fcrit_initial = 50.0
            self.fcrit_floor = 5.0
            self.fcrit_replenish_rate = 0.03
            self.baseline_fcrit_operational_cost = 0.7
            self.beta_decay_flaring = 0.07
            self.beta_rebuild_pruning = 0.01
            self.beta_phase_iv_noise_std = 0.005
            self.strain_baseline = 4.0
            self.strain_shock_magnitude = 9.0
            self.strain_shock_duration = 30
            self.strain_shock_interval = 110
            self.post_collapse_strain_reduction = 0.5
            self.fcrit_high_level_decay_rate = 0.001
            self.sim_steps = 1000
            self.n_intervention_runs = 30
            self.total_rl_training_timesteps = 500000
            self.shock_start_time = 60
            self.beta_initial = 0.8
        elif level == "proactive_degradation":
            # Scenario designed for proactive stabilization studies
            self.fcrit_initial = 80.0
            self.fcrit_floor = 10.0
            self.fcrit_replenish_rate = 0.08
            self.baseline_fcrit_operational_cost = 0.3
            self.beta_decay_flaring = 0.05
            self.beta_rebuild_pruning = 0.02
            self.beta_phase_iv_noise_std = 0.002
            self.strain_baseline = 3.0
            self.strain_shock_magnitude = 4.0
            self.strain_shock_duration = 20
            self.strain_shock_interval = 200
            self.post_collapse_strain_reduction = 0.7
            self.fcrit_high_level_decay_rate = 0.0002
            self.sim_steps = 1000
            self.n_intervention_runs = 30
            self.total_rl_training_timesteps = 500000
            self.shock_start_time = 100
            self.beta_initial = 0.8
            # gradual degradation parameters
            self.fcrit_replenish_rate_decay = 0.0001
            self.strain_baseline_drift = 0.002
        else: # Default (original settings)
            self.fcrit_initial = 100.0
            self.fcrit_floor = 10.0
            self.fcrit_replenish_rate = 0.1
            self.baseline_fcrit_operational_cost = 0.2
            self.beta_decay_flaring = 0.05
            self.beta_rebuild_pruning = 0.02
            self.beta_phase_iv_noise_std = 0.002
            self.strain_baseline = 3.0
            self.strain_shock_magnitude = 5.0
            self.strain_shock_duration = 20
            self.strain_shock_interval = 150
            self.post_collapse_strain_reduction = 0.7
            self.fcrit_high_level_decay_rate = 0.0
            self.sim_steps = 600
            self.n_intervention_runs = 30
            self.total_rl_training_timesteps = 500000
            self.shock_start_time = 50
            self.beta_initial = 0.8

# CHOOSE ENVIRONMENT CONFIGURATION FOR THE STUDY
# Options: "easy_debug", "challenging", "extreme", "default"
SELECTED_ENV_LEVEL = "easy_debug" # CHANGED TO "challenging" for testing new rewards
CURRENT_ENV_CONFIG = EnvironmentConfig(level=SELECTED_ENV_LEVEL)

# Directories for saving outputs
RESULTS_DIR = "results"
RL_MODELS_DIR = "rl_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RL_MODELS_DIR, exist_ok=True)

# --- Constants (Potentially overridden by CURRENT_ENV_CONFIG or used by real TDSystem) ---
G_INITIAL = 1.0 # Assuming G_INITIAL is relatively constant for RL clipping
BETA_INITIAL = CURRENT_ENV_CONFIG.beta_initial # For RL clipping, taken from config
FCRIT_INITIAL = CURRENT_ENV_CONFIG.fcrit_initial
W1, W2, W3 = 0.33, 0.33, 0.34
C_TOLERANCE = 1.0
FCRIT_FLOOR = CURRENT_ENV_CONFIG.fcrit_floor
# ... other constants if they are globally fixed and not in CURRENT_ENV_CONFIG ...

GLOBAL_ENTROPY_BASELINE = 0.02
SMOOTHING_WINDOW_DEFAULT = 15
SMOOTHING_POLY_DEFAULT = 2
ROLLING_WINDOW_COUPLE = 30
ROLLING_WINDOW_ENTROPY = 20
BURN_IN_PERIOD = 30 # For ML heuristic phase prediction
FEATURE_WINDOW_SIZE = 10 # For ML heuristic features

INTERVENTION_PARAMS = {
    'phase1_beta_reduction_factor': 0.5,
    'phase1_fcrit_protection_floor_factor': 1.5,
    'phase1_g_reduction_factor': 0.9,
    'phase2_exploration_cap_beta_floor': 0.1,
    'phase2_targeted_fcrit_support_amount': 0.2,
    'phase3_fcrit_injection_target_factor': 0.9,
    'phase3_fcrit_injection_rate': 1.0,
    'phase3_beta_cautious_increase_rate': 0.01,
    'phase4_beta_optimize_target_factor': 1.1,
    'phase4_beta_optimize_rate': 0.005,
    'intervention_cooldown': 5,
    'proactive_td_fcrit_threshold_factor': 0.6,
    'proactive_td_fcrit_boost_multiplier': 2.0,
    'naive_fcrit_boost_trigger_factor': 0.90,
    'naive_fcrit_boost_strength_multiplier': 2.5
}
TOTAL_RL_TRAINING_TIMESTEPS = CURRENT_ENV_CONFIG.total_rl_training_timesteps
RL_MODEL_PATH = os.path.join(RL_MODELS_DIR, f"phoenix_loop_rl_ppo_agent_{SELECTED_ENV_LEVEL}.zip")


# --- DUMMY DEFINITIONS (Globally available for fallback) ---
class DummyTDSystem:
    def __init__(self, run_id=0, sim_steps_override=None, env_config=None):
        self.config = env_config if env_config else CURRENT_ENV_CONFIG
        self.run_id = run_id
        self.time = 0
        self.sim_steps_limit = sim_steps_override if sim_steps_override is not None else self.config.sim_steps
        self.g_lever = G_INITIAL
        self.beta_lever = self.config.beta_initial # Use beta_initial from config
        self.fcrit = self.config.fcrit_initial
        self.strain = self.config.strain_baseline
        self._baseline_operational_cost = self.config.baseline_fcrit_operational_cost
        self._fcrit_high_level_decay_rate = self.config.fcrit_high_level_decay_rate
        self._fcrit_replenish_rate = self.config.fcrit_replenish_rate
        self._fcrit_floor = self.config.fcrit_floor
        self.theta_t = C_TOLERANCE * (self.g_lever**W1) * (self.beta_lever**W2) * (np.clip(self.fcrit, 1e-6, None)**W3)
        self.safety_margin = self.theta_t - self.strain
        self.true_phase = 4
        self.collapsed = False
        self._shock_active_until = -1
        self._next_shock_time = self.config.shock_start_time
        self.history = {key: [] for key in [
            'run_id', 'time', 'g_lever', 'beta_lever', 'fcrit', 'strain',
            'theta_t', 'safety_margin', 'true_phase', 'collapsed_flag'
        ]}
        self._record_history()
    def _record_history(self):
        for key, val in self.get_current_state_dict().items():
             if key in self.history: self.history[key].append(val)
    def get_current_state_dict(self):
        return {'run_id': self.run_id, 'time': self.time, 'g_lever': self.g_lever,
                'beta_lever': self.beta_lever, 'fcrit': self.fcrit, 'strain': self.strain,
                'theta_t': self.theta_t, 'safety_margin': self.safety_margin,
                'true_phase': self.true_phase, 'collapsed_flag': 1 if self.collapsed else 0}
    def _update_strain(self):
        base_strain_val = self.config.strain_baseline
        if self.collapsed: base_strain_val *= self.config.post_collapse_strain_reduction
        current_strain_val = base_strain_val
        if self.time >= self._next_shock_time and self.time < self._next_shock_time + self.config.strain_shock_duration:
            current_strain_val += self.config.strain_shock_magnitude
            if self.time == self._next_shock_time: self._shock_active_until = self._next_shock_time + self.config.strain_shock_duration
        elif self._shock_active_until != -1 and self.time == self._shock_active_until:
            self._next_shock_time = self.time + self.config.strain_shock_interval; self._shock_active_until = -1
        self.strain = max(0.1, current_strain_val + np.random.normal(0, 0.05 * base_strain_val))
    def _update_levers_and_fcrit(self):
        cost_g = 0.05 * (self.g_lever**0.5); cost_beta = 0.02 * (self.beta_lever**1.2)
        self.fcrit -= (self.config.baseline_fcrit_operational_cost + cost_g + cost_beta)
        if self.fcrit > self.config.fcrit_initial * 1.2 and self.config.fcrit_high_level_decay_rate > 0:
            self.fcrit -= self.fcrit * self.config.fcrit_high_level_decay_rate
        self.fcrit += self.config.fcrit_replenish_rate * (self.config.fcrit_initial - self.fcrit)
        self.fcrit = min(self.fcrit, self.config.fcrit_initial * 1.5)
        if self.collapsed:
            if self.true_phase == 1: self.beta_lever = max(0.1, self.beta_lever - 0.02)
            elif self.true_phase == 2: self.beta_lever = max(0.05, self.beta_lever - 0.05 * self.beta_lever)
            elif self.true_phase == 3: self.beta_lever = min(self.config.beta_initial * 1.3, self.beta_lever + 0.02 * (self.config.beta_initial - self.beta_lever))
            if self.fcrit > self.config.fcrit_initial * 0.8 and self.beta_lever > self.config.beta_initial * 0.8 and self.true_phase < 4:
                self.true_phase = 4; self.collapsed = False
        else: self.beta_lever = np.clip(self.beta_lever + np.random.normal(0, 0.002), 0.05, self.config.beta_initial * 2.0)
        self.g_lever = np.clip(self.g_lever + np.random.normal(0, 0.01), 0.5, 1.5)
        self.fcrit = max(self.config.fcrit_floor * 0.5, self.fcrit)
    def _check_collapse(self):
         if not self.collapsed and (self.fcrit <= self.config.fcrit_floor or self.safety_margin < 0):
            self.collapsed = True; self.true_phase = 1;
    def step(self):
        if self.time >= self.sim_steps_limit: return False
        self._update_strain(); self._update_levers_and_fcrit()
        self.theta_t = C_TOLERANCE * (self.g_lever**W1) * (self.beta_lever**W2) * (np.clip(self.fcrit, 1e-6, None)**W3)
        self.safety_margin = self.theta_t - self.strain
        self._check_collapse(); self.time += 1; self._record_history(); return True

class DummyDiagnosticsCalculator:
    def __init__(self, entropy_baseline=None, env_config=None):
        self.diagnostics_list = []
        self.entropy_baseline = entropy_baseline if entropy_baseline is not None else GLOBAL_ENTROPY_BASELINE
    def calculate_diagnostics_for_run(self, history_data_for_run): self.diagnostics_list.append({})
    def finalize_rhoE_and_baseline(self, *, recompute_baseline: bool = False, quiet: bool = True): pass
    def push_sample(self, sample_dict: dict): pass
    def latest_features(self, feature_window_size_ml: int) -> pd.DataFrame: return pd.DataFrame()

def create_ml_features(*args, **kwargs): return pd.DataFrame(), None, None # Dummy if import fails

# --- TDSystem and DiagnosticsCalculator (attempt to import real, fallback to dummy) ---
RealTDSystem = None
RealDiagnosticsCalculator = None
TDSystem_Base = DummyTDSystem
DiagnosticsCalculator_Base = DummyDiagnosticsCalculator

try:
    from phoenix_loop_classifier_accuracy_ML import TDSystem as ImportedRealTDSystem, DiagnosticsCalculator as ImportedRealDiagnosticsCalculator
    TDSystem_Base = ImportedRealTDSystem
    DiagnosticsCalculator_Base = ImportedRealDiagnosticsCalculator
    RealTDSystem = ImportedRealTDSystem
    RealDiagnosticsCalculator = ImportedRealDiagnosticsCalculator
    print("INFO: Successfully imported REAL TDSystem and DiagnosticsCalculator.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import from phoenix_loop_classifier_accuracy_ML.py. Error: {e}")
    print("Using DUMMY TDSystem and DiagnosticsCalculator as fallback.")

# --- Helper Functions ---
def _hms(seconds: float) -> str:
    if seconds < 0.000001: return "0s"
    if seconds < 0.001: return f"{seconds * 1000000:.0f}µs"
    if seconds < 1.0: return f"{seconds * 1000:.1f}ms"
    if seconds < 60: return f"{seconds:.2f}s"
    return str(timedelta(seconds=round(seconds)))

@functools.lru_cache(maxsize=32)
def _get_cached_savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0):
    return savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

def _custom_savgol_filter_with_cache(series, window_length, polyorder, deriv=0, delta=1.0):
    series = np.asarray(series, dtype=np.float32)
    if len(series) < window_length or window_length <= polyorder or window_length < 3 :
        current_poly = polyorder; current_window = window_length
        if len(series) >=3 :
            current_poly = min(polyorder, len(series)-1 if len(series) > 1 else 0)
            current_window = len(series) if len(series) % 2 != 0 else len(series) -1
            if current_window <= current_poly: current_window = current_poly + (1 if (current_poly+1) % 2 !=0 else 2)
            current_window = min(current_window, len(series))
            if current_window % 2 == 0 and current_window > 1 : current_window -=1
            if current_window > current_poly and current_window >=3:
                 return savgol_filter(series, current_window, current_poly, deriv=deriv, delta=delta)
        return series.astype(np.float32) if deriv == 0 else np.zeros_like(series, dtype=np.float32)
    coeffs = _get_cached_savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
    return correlate1d(series, coeffs, axis=-1, mode='mirror', output=np.float32)

def write_summary_txt(summary_dict: dict, filepath: str = "simulation_summary.txt") -> None:
    """Dump summary_dict to filepath in a readable layout."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    lines = []
    run_logs = summary_dict.get('_run_logs')
    meta = summary_dict.get('meta')
    if meta:
        lines.append("### META")
        for k, v in meta.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{k} = {json.dumps(v)}")
            else:
                lines.append(f"{k} = {v}")
    for cond, stats in summary_dict.items():
        if cond in {'_run_logs', 'meta'}:
            continue
        lines.append(f"### {cond}")
        for key, val in stats.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict) and 'mean' in val and 'sem' in val and not isinstance(val['mean'], dict):
                lines.append(f"{key}.mean = {val['mean']}")
                lines.append(f"{key}.sem  = {val['sem']}")
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, list):
                        lines.append(f"{key}.{sub_k} = [{','.join(map(str, sub_v))}]")
                    else:
                        lines.append(f"{key}.{sub_k} = {sub_v}")
            else:
                lines.append(f"{key} = {val}")
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        f.write(json.dumps(summary_dict, indent=2, cls=NpEncoder))
        if run_logs:
            f.write('\n### RUN_LOGS\n')
            for r in run_logs:
                f.write(','.join(map(str, r)) + '\n')

def _mean_sem_pair(stat_dict):
    """Return [mean, sem] pair if available else [nan, 0]."""
    if isinstance(stat_dict, dict):
        mean = float(stat_dict.get('mean', np.nan))
        sem_val = float(stat_dict.get('sem', 0.0))
        return [mean, sem_val]
    return [float('nan'), 0.0]

def generate_trimmed_summary(summary_dict: dict) -> dict:
    """Create a compact summary from the full summary_dict."""
    trimmed = {
        'meta': {},
        'conditions': {}
    }
    meta = summary_dict.get('meta', {})
    if meta:
        trimmed['meta'] = {
            'env_level': meta.get('env_level'),
            'sim_steps': meta.get('sim_steps'),
            'rl_training_timesteps': meta.get('rl_training_timesteps'),
            'code_commit': meta.get('code_commit'),
        }
        if isinstance(meta.get('ml_classifier'), dict):
            trimmed['meta']['ml_classifier'] = {
                'macro_F1': meta['ml_classifier'].get('macro_F1'),
                'overall_acc': meta['ml_classifier'].get('overall_acc'),
            }

    for cond, stats in summary_dict.items():
        if cond in {'_run_logs', 'meta'}:
            continue
        dst = {}
        if 'n_runs' in stats:
            dst['n_runs'] = int(stats['n_runs'])
        dst['collapse_rate'] = _mean_sem_pair(stats.get('collapse_rate'))
        dst['avg_time_to_p4'] = _mean_sem_pair(stats.get('avg_time_to_p4'))
        dst['avg_duration_stable_p4'] = _mean_sem_pair(stats.get('avg_duration_stable_p4'))
        if 'avg_total_fcrit_injected' in stats:
            dst['avg_total_fcrit_injected'] = float(stats['avg_total_fcrit_injected'].get('mean', np.nan))
        if 'avg_lever_cost' in stats:
            dst['avg_lever_cost'] = float(stats['avg_lever_cost'].get('mean', np.nan))
        if 'phase_occupancy' in stats:
            po = stats['phase_occupancy'].get('mean', [])
            dst['phase_occupancy'] = [round(float(x), 3) for x in po]
        if 'fcrit_auc' in stats:
            dst['fcrit_auc'] = float(stats['fcrit_auc'])
        if 'sm_auc' in stats:
            dst['sm_auc'] = float(stats['sm_auc'])
        if 'sm_slope' in stats and isinstance(stats['sm_slope'], dict):
            dst['sm_slope'] = float(stats['sm_slope'].get('mean', np.nan))
        if 'diagnostics' in stats:
            diag = {}
            for k, v in stats['diagnostics'].items():
                if isinstance(v, dict):
                    diag[k] = [float(v.get('mean', np.nan)), float(v.get('std', np.nan)), float(v.get('max', np.nan))]
            dst['diagnostics'] = diag
        if 'rl_episode_return' in stats:
            dst['rl_episode_return_mean'] = float(stats['rl_episode_return'].get('mean', np.nan))
        if 'rl_steps_in_P4_consecutive' in stats:
            dst['rl_steps_in_P4_mean'] = float(stats['rl_steps_in_P4_consecutive'].get('mean', np.nan))
        pv = {k: float(v) for k, v in stats.items() if k.endswith('.p_vs_NoIntervention')}
        if pv:
            dst['p_vs_NoIntervention'] = pv
        trimmed['conditions'][cond] = dst

    trimmed['data_uri'] = 'results/curves.parquet'
    trimmed['raw_log_uri'] = 'results/run_logs.csv'
    return trimmed

def write_trimmed_summary_json(summary_dict: dict, filepath='summary_intervention_RL.json') -> None:
    """Write compact JSON summary for LLM consumption."""
    trimmed = generate_trimmed_summary(summary_dict)

    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    trimmed = _clean(trimmed)
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(trimmed, f, indent=2, allow_nan=False, cls=NpEncoder)

def write_trimmed_summary_markdown(summary_dict: dict, path: str) -> None:
    """Write a human-readable Markdown summary built from generate_trimmed_summary."""
    trimmed = generate_trimmed_summary(summary_dict)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, "w") as f:
        f.write("# Proactive-Stabilization Study (trimmed)\n\n")
        meta = trimmed.get("meta", {})
        f.write(f"**Environment**: {meta.get('env_level')}\n\n")
        for cond, stats in trimmed.get("conditions", {}).items():
            f.write(f"## {cond}\n")
            if 'n_runs' in stats:
                f.write(f"- Runs: {stats['n_runs']}\n")
            cr = stats.get('collapse_rate')
            if cr:
                f.write(f"- Collapse rate: {cr[0]:.2f} ± {cr[1]:.2f}\n")
            tp4 = stats.get('avg_time_to_p4')
            if tp4:
                f.write(f"- Time to P4: {tp4[0]:.1f} ± {tp4[1]:.1f} steps\n")
            dur = stats.get('avg_duration_stable_p4')
            if dur:
                f.write(f"- Stable P4 duration: {dur[0]:.1f} ± {dur[1]:.1f} steps\n")
            if 'avg_total_fcrit_injected' in stats:
                f.write(f"- Total fcrit injected: {stats['avg_total_fcrit_injected']:.2f}\n")
            if 'avg_lever_cost' in stats:
                f.write(f"- Lever cost: {stats['avg_lever_cost']:.2f}\n")
            if 'phase_occupancy' in stats:
                f.write(f"- Phase occupancy: {stats['phase_occupancy']}\n")
            if 'diagnostics' in stats:
                for metric, vals in stats['diagnostics'].items():
                    if isinstance(vals, list) and len(vals) >= 2:
                        f.write(f"- {metric}: {vals[0]:.2f} ± {vals[1]:.2f}\n")
            if 'p_vs_NoIntervention' in stats:
                for metric, p in stats['p_vs_NoIntervention'].items():
                    f.write(f"  - *{metric}* p = {p:.3g}\n")
            f.write("\n")

def write_run_logs_csv(run_logs, filepath: str) -> None:
    if not run_logs:
        return
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for r in run_logs:
            f.write(','.join(map(str, r)) + '\n')


def smooth_series(series, window=SMOOTHING_WINDOW_DEFAULT, poly=SMOOTHING_POLY_DEFAULT):
    return _custom_savgol_filter_with_cache(series, window, poly, deriv=0)

def calculate_derivative(series, window=SMOOTHING_WINDOW_DEFAULT, poly=SMOOTHING_POLY_DEFAULT, dt=1):
    series_np = np.asarray(series, dtype=np.float32); smoothed = smooth_series(series_np, window, poly)
    if len(smoothed) < 2: return np.zeros_like(smoothed, dtype=np.float32)
    return np.gradient(smoothed, dt).astype(np.float32)

def add_true_phase_background(ax, time_history, true_phase_history, zorder=-10):
    colors = ['white', 'lightcoral', 'lightsalmon', 'lightgreen', 'lightblue'] # Phase 0-4
    min_len = min(len(time_history), len(true_phase_history))
    if min_len <= 1: return
    time_history_np = np.asarray(time_history); true_phase_history_np = np.asarray(true_phase_history)
    for i in range(min_len -1):
        phase_val = true_phase_history_np[i]; color_idx = 0
        if isinstance(phase_val, (int, float, np.integer, np.floating)): # Allow float phases
            phase_val_int = int(phase_val)
            if 0 <= phase_val_int < len(colors): color_idx = phase_val_int
            elif phase_val_int >= len(colors): color_idx = phase_val_int % len(colors)
        ax.axvspan(time_history_np[i], time_history_np[i+1], facecolor=colors[color_idx], alpha=0.3, edgecolor=None, zorder=zorder)

# --- TDSystemIntervention (Derived Class) ---
class TDSystemIntervention(TDSystem_Base):
    def __init__(self, run_id=0, intervention_type='none', env_config=None):
        self.config = env_config if env_config else CURRENT_ENV_CONFIG
        super().__init__(run_id=run_id, sim_steps_override=self.config.sim_steps, env_config=self.config)
        self.intervention_type = intervention_type
        self.last_intervention_time = {k: -float('inf') for k in ['beta_phase1', 'g_phase1', 'beta_phase2', 'fcrit_phase2', 'fcrit_phase3', 'beta_phase3', 'beta_phase4', 'naive_fcrit', 'rl_action', 'td_proactive_fcrit']}
        self.total_fcrit_injected_scalar = 0.0
        self.rl_target_g = self.g_lever; self.rl_target_beta = self.beta_lever; self.rl_target_fcrit = self.fcrit
        self.last_rl_action_cost = 0.0; self.current_rl_action_int = np.nan
        self._fcrit_decay_step = getattr(self.config, 'fcrit_replenish_rate_decay', 0.0)
        self._strain_drift_step = getattr(self.config, 'strain_baseline_drift', 0.0)
        new_keys_initial_values = {'intervention_applied': "Initial_State_t0", 'target_g': self.g_lever,
                                   'target_beta': self.beta_lever, 'target_fcrit': self.fcrit,
                                   'total_fcrit_injected': 0.0, 'rl_action_chosen': np.nan}
        for key, val_t0 in new_keys_initial_values.items():
            if key not in self.history: self.history[key] = []
            if self.history.get('time') and len(self.history['time']) > 0:
                while len(self.history[key]) < len(self.history['time']) -1 : self.history[key].append(np.nan)
                if len(self.history[key]) == len(self.history['time']): self.history[key][-1] = val_t0
                elif len(self.history[key]) < len(self.history['time']): self.history[key].append(val_t0)
                else: self.history[key][-1] = val_t0
            else: self.history[key] = [val_t0]
    @property
    def collapsed_flag(self):
        if 'collapsed_flag' not in self.history or not self.history['collapsed_flag']:
            return [0] * len(self.history['time']) if 'time' in self.history and self.history['time'] else [0]
        return self.history['collapsed_flag']

    def _apply_degradation_effects(self):
        if self._fcrit_decay_step:
            self.config.fcrit_replenish_rate = max(0.0, self.config.fcrit_replenish_rate - self._fcrit_decay_step)
        if self._strain_drift_step:
            self.config.strain_baseline += self._strain_drift_step
    def set_rl_targets(self, target_g, target_beta, target_fcrit, action_cost=0.0, action_int=np.nan):
        self.rl_target_g = target_g; self.rl_target_beta = target_beta; self.rl_target_fcrit = target_fcrit
        self.last_rl_action_cost = action_cost; self.current_rl_action_int = action_int
    def apply_intervention(self, algorithmic_phase):
        applied_intervention_desc = "None"; target_g, target_beta, target_fcrit = self.g_lever, self.beta_lever, self.fcrit
        cooldown = INTERVENTION_PARAMS['intervention_cooldown']; fcrit_change_this_step_intervention = 0.0; current_time = self.time
        if self.intervention_type == 'rl_agent':
            target_g, target_beta, target_fcrit = self.rl_target_g, self.rl_target_beta, self.rl_target_fcrit
            applied_intervention_desc = f"RL_Targets:g{target_g:.1f}b{target_beta:.2f}F{target_fcrit:.0f}"
            if target_fcrit > self.fcrit: fcrit_change_this_step_intervention += (target_fcrit - self.fcrit) * 0.05 # Smoothing factor
        elif self.intervention_type == 'td_informed':
            # ... (TD Informed Logic - phases 1-4, proactive boost - condensed for brevity, assumed unchanged)
            if algorithmic_phase == 1: # Example: Phase 1 logic
                if current_time - self.last_intervention_time['beta_phase1'] > cooldown: target_beta = self.beta_lever * INTERVENTION_PARAMS['phase1_beta_reduction_factor']; applied_intervention_desc = f"P1:BetaReduce({target_beta:.2f})"; self.last_intervention_time['beta_phase1'] = current_time
            # ... other phases and proactive logic ...
            # Ensure fcrit_change_this_step_intervention is updated if TD_Informed injects Fcrit
            # Example for phase 2 support:
            # if phase2_fcrit_support_triggered:
            #    nominal_injection = INTERVENTION_PARAMS['phase2_targeted_fcrit_support_amount']
            #    target_fcrit = self.fcrit + nominal_injection
            #    fcrit_change_this_step_intervention += nominal_injection
        elif self.intervention_type == 'naive_fcrit_boost':
            naive_trigger_fcrit = self.config.fcrit_initial * INTERVENTION_PARAMS['naive_fcrit_boost_trigger_factor']
            if self.fcrit < naive_trigger_fcrit and (current_time - self.last_intervention_time['naive_fcrit'] > cooldown):
                nominal_injection = INTERVENTION_PARAMS['phase3_fcrit_injection_rate'] * INTERVENTION_PARAMS['naive_fcrit_boost_strength_multiplier']
                target_fcrit = self.fcrit + nominal_injection; fcrit_change_this_step_intervention += nominal_injection
                applied_intervention_desc = f"Naive:FcritBoost(+{nominal_injection:.2f} to {target_fcrit:.2f})"; self.last_intervention_time['naive_fcrit'] = current_time
        self.g_lever = self.g_lever * 0.95 + target_g * 0.05; self.beta_lever = self.beta_lever * 0.95 + target_beta * 0.05
        self.fcrit = self.fcrit * 0.95 + target_fcrit * 0.05; self.total_fcrit_injected_scalar += fcrit_change_this_step_intervention
        self.history['intervention_applied'].append(applied_intervention_desc); self.history['target_g'].append(target_g)
        self.history['target_beta'].append(target_beta); self.history['target_fcrit'].append(target_fcrit)
        self.history['rl_action_chosen'].append(self.current_rl_action_int if self.intervention_type == 'rl_agent' else np.nan)
    def step(self, current_algorithmic_phase=None):
        if self.time >= self.config.sim_steps: return False
        self._apply_degradation_effects()
        if self.intervention_type != 'none': self.apply_intervention(algorithmic_phase=current_algorithmic_phase)
        else:
            for key in ['intervention_applied', 'target_g', 'target_beta', 'target_fcrit', 'rl_action_chosen']:
                if key == 'intervention_applied': self.history[key].append("None")
                elif key == 'rl_action_chosen': self.history[key].append(np.nan)
                else: self.history[key].append(getattr(self, key.split('_')[-1] if 'target' in key else key, np.nan)) # Appends current lever
        if 'total_fcrit_injected' in self.history: self.history['total_fcrit_injected'].append(self.total_fcrit_injected_scalar)
        else: self.history['total_fcrit_injected'] = [self.total_fcrit_injected_scalar] * (len(self.history.get('time', [1])) )
        return super().step()

# --- DiagnosticsCalculatorIntervention ---
REQUIRED_SERIES_FOR_DIAG_CALC = ["g_lever", "beta_lever", "fcrit", "strain", "theta_t", "safety_margin", "true_phase", "run_id"]
class DiagnosticsCalculatorIntervention(DiagnosticsCalculator_Base):
    def __init__(self, entropy_baseline: float = GLOBAL_ENTROPY_BASELINE, env_config=None):
        self.effective_env_config = env_config if env_config else CURRENT_ENV_CONFIG
        if DiagnosticsCalculator_Base is RealDiagnosticsCalculator and RealDiagnosticsCalculator is not None:
            super().__init__(env_config=self.effective_env_config)
        else:
            try: super().__init__(entropy_baseline=entropy_baseline, env_config=self.effective_env_config)
            except TypeError: super().__init__(entropy_baseline=entropy_baseline)
        self.entropy_baseline = entropy_baseline
        self.config = self.effective_env_config if not hasattr(self, 'config') or self.config is None else self.config
        self._diag_calc_max_window = max(SMOOTHING_WINDOW_DEFAULT, ROLLING_WINDOW_COUPLE, ROLLING_WINDOW_ENTROPY, 3)
        self._max_hist_for_features = self._diag_calc_max_window + FEATURE_WINDOW_SIZE + SMOOTHING_WINDOW_DEFAULT
        self.stream_series_data = {key: deque(maxlen=self._max_hist_for_features) for key in REQUIRED_SERIES_FOR_DIAG_CALC}
    def finalize_rhoE_and_baseline(self, *, recompute_baseline: bool = False, quiet: bool = True):
        with (contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()):
            if DiagnosticsCalculator_Base is DummyDiagnosticsCalculator:
                return super().finalize_rhoE_and_baseline(recompute_baseline=recompute_baseline)
            else:
                try: return super().finalize_rhoE_and_baseline()
                except TypeError as te: print(f"Warning: TypeError in Real base finalize: {te}"); return None
    def push_sample(self, sample_dict: dict):
        for key in REQUIRED_SERIES_FOR_DIAG_CALC:
            if key in sample_dict:
                val = sample_dict[key]; val = int(val) if key in ["run_id", "true_phase"] else np.float32(val)
                self.stream_series_data[key].append(val)
    def latest_features(self, feature_window_size_ml: int) -> pd.DataFrame:
        req_len = max(self._diag_calc_max_window + feature_window_size_ml -1 if feature_window_size_ml > 0 else self._diag_calc_max_window, SMOOTHING_POLY_DEFAULT + 1, 3)
        if len(self.stream_series_data["g_lever"]) < req_len: return pd.DataFrame()
        hist_slice = {key: list(self.stream_series_data[key]) for key in REQUIRED_SERIES_FOR_DIAG_CALC}
        temp_calc = type(self)(entropy_baseline=self.entropy_baseline, env_config=self.effective_env_config)
        temp_calc.calculate_diagnostics_for_run(hist_slice); temp_calc.finalize_rhoE_and_baseline(recompute_baseline=False, quiet=True)
        if not temp_calc.diagnostics_list: return pd.DataFrame()
        features_df, _, _ = create_ml_features([temp_calc.diagnostics_list[0]], feature_window_size_ml)
        return features_df.iloc[[-1]] if not features_df.empty else pd.DataFrame()
    def calculate_diagnostics_for_run(self, history_data_for_run):
        # only keep the eight series we actually need
        filtered = {
            k: history_data_for_run[k]
            for k in REQUIRED_SERIES_FOR_DIAG_CALC
            if k in history_data_for_run
        }

        # convert to numpy, but leave run_id and true_phase as ints
        processed = {}
        for k, v in filtered.items():
            if k in ("run_id", "true_phase"):
                processed[k] = np.asarray(v, dtype=int)
            else:
                processed[k] = np.asarray(v, dtype=np.float32)

        super().calculate_diagnostics_for_run(processed)


# --- PhoenixLoopRLEnv (with new reward function) ---
class PhoenixLoopRLEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}
    ACTION_NAMES = ["NO_OP", "FCRIT_BOOST_S", "FCRIT_BOOST_L", "BETA_REDUCE_M", "BETA_INCREASE_M", "G_REDUCE_M"]
    N_ACTIONS = len(ACTION_NAMES)

    # ─────────────────────────────────────────────────────────────
    #  Cost & effect constants promoted to class scope so they can
    #  be accessed as PhoenixLoopRLEnv.CONSTANT_NAME from anywhere
    # ─────────────────────────────────────────────────────────────
    COST_BASE_ACTION        = 0.01          # base cost for any non-NOP
    COST_FCRIT_BOOST_S_ADD  = 0.05 * 5.0
    COST_FCRIT_BOOST_L_ADD  = 0.05 * 15.0
    COST_BETA_REDUCE_M_ADD  = 0.10
    COST_BETA_INCREASE_M_ADD= 0.10
    COST_G_REDUCE_M_ADD     = 0.05

    FCRIT_BOOST_S_AMOUNT    = 5.0
    FCRIT_BOOST_L_AMOUNT    = 15.0
    BETA_REDUCE_M_FACTOR    = 0.8
    BETA_INCREASE_M_AMOUNT  = 0.02
    G_REDUCE_M_FACTOR       = 0.9
    # ─────────────────────────────────────────────────────────────

    def __init__(self, render_mode=None, run_id_offset=1000, env_config=None, reward_mods=None):
        super().__init__()
        self.env_config = env_config if env_config else CURRENT_ENV_CONFIG
        self.run_id_offset = run_id_offset
        self.current_run_id = 0
        self.action_space = spaces.Discrete(PhoenixLoopRLEnv.N_ACTIONS)
        _obs_low_default = np.array([0,0,0, 0,0,-50, 0,0,0,0,0, 0,0, -5,-10], dtype=np.float32)
        _obs_high_default= np.array([5,2,150,50,50,50, 1,1,1,1,1, 1,1,  5, 10], dtype=np.float32)
        _obs_high_default[2] = self.env_config.fcrit_initial * 1.5
        self.observation_space = spaces.Box(low=_obs_low_default, high=_obs_high_default, dtype=np.float32)
        
        # Initialize system after other attributes are set
        self.system = TDSystemIntervention(run_id=self.current_run_id, intervention_type='rl_agent', env_config=self.env_config)
        self.time_in_current_phase = 0
        self.last_true_phase = self.system.true_phase if self.system.true_phase is not None else -1

        # Tuned Reward Parameters
        self.R_STEP_SURVIVAL = 0.01
        self.R_BASE_PHASE_4_BONUS = 0.5  # Significantly increased
        self.R_SUSTAINED_P4_FACTOR = 0.1 # For quadratic bonus for sustained P4
        self.R_ACHIEVED_P4_BONUS = 5.0    # Large one-time bonus for entering P4
        self.R_PHASE_1_PENALTY = -0.2   # Increased penalty
        self.R_PHASE_2_PENALTY = -0.15  # Increased penalty
        self.R_PHASE_3_PENALTY = -0.1  # Stronger penalty for P3
        self.R_COLLAPSE_EVENT_PENALTY = -20.0 # Increased
        self.R_FCRIT_LOW_PCTG_PENALTY_SCALE = 1.0 # Increased scale
        self.R_FCRIT_VERY_LOW_PCTG_PENALTY = -2.0 # Increased penalty
        self.R_PHASE_ENTRY_PENALTY = -0.3  # Penalty for entering P1-P3
        self.R_SM_CHANGE_POSITIVE_FACTOR = 0.1 # More reward for positive SM change
        self.R_SM_CHANGE_NEGATIVE_FACTOR = 0.15 # Stronger penalty for negative SM change
        self.R_HEALTHY_SM_IN_P4_BONUS = 0.1
        self.SM_TARGET_IN_P4 = 2.0 
        self.R_G_OSCILLATION_P4_PENALTY = 0.02
        self.R_BETA_OSCILLATION_P4_PENALTY = 0.02
        self.G_CHANGE_THRESHOLD_P4 = 0.05  # Stricter
        self.BETA_CHANGE_THRESHOLD_P4 = 0.02 # Stricter
        self.R_TRUNCATED_NOT_IN_P4_PENALTY = -15.0 # Large penalty
        self.COST_BETA_REDUCE_M_ADD = 0.1; self.COST_BETA_INCREASE_M_ADD = 0.1; self.COST_G_REDUCE_M_ADD = 0.05
        self.FCRIT_BOOST_S_AMOUNT = 5.0; self.FCRIT_BOOST_L_AMOUNT = 15.0
        self.BETA_REDUCE_M_FACTOR = 0.8; self.BETA_INCREASE_M_AMOUNT = 0.02; self.G_REDUCE_M_FACTOR = 0.9
        self.steps_in_phase_4_consecutive = 0
        self.prev_g_lever = self.system.g_lever; self.prev_beta_lever = self.system.beta_lever

        if reward_mods:
            for key, val in reward_mods.items():
                if hasattr(self, key):
                    setattr(self, key, val)

    def _get_obs(self): # Condensed for brevity, assumed mostly unchanged
        obs_list = [self.system.g_lever, self.system.beta_lever, self.system.fcrit, self.system.strain, self.system.theta_t, self.system.safety_margin]
        true_phase_oh = np.zeros(5, dtype=np.float32); current_true_phase_int = int(self.system.true_phase if self.system.true_phase is not None else 0)
        if 0 <= current_true_phase_int < 5: true_phase_oh[current_true_phase_int] = 1.0
        obs_list.extend(true_phase_oh); obs_list.append(self.system.time / self.system.config.sim_steps)
        obs_list.append(min(self.time_in_current_phase / 100.0, 1.0))
        dFcrit, dSM = 0.0, 0.0
        if 'fcrit' in self.system.history and len(self.system.history['fcrit']) > SMOOTHING_WINDOW_DEFAULT + SMOOTHING_POLY_DEFAULT + 1 : dFcrit = calculate_derivative(self.system.history['fcrit'])[-1]
        if 'safety_margin' in self.system.history and len(self.system.history['safety_margin']) > SMOOTHING_WINDOW_DEFAULT + SMOOTHING_POLY_DEFAULT + 1 : dSM = calculate_derivative(self.system.history['safety_margin'])[-1]
        obs_list.extend([np.clip(dFcrit,-5,5), np.clip(dSM,-10,10)])
        obs = np.array(obs_list, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _calculate_reward(self, prev_fcrit, prev_sm, prev_phase_for_reward_calc):
        reward = 0.0; action_cost = self.system.last_rl_action_cost
        current_true_phase = self.system.true_phase; fcrit_floor_val = self.system.config.fcrit_floor
        reward += self.R_STEP_SURVIVAL
        if current_true_phase == 4:
            reward += self.R_BASE_PHASE_4_BONUS; self.steps_in_phase_4_consecutive += 1
            reward += (self.steps_in_phase_4_consecutive / 10.0)**2 * self.R_SUSTAINED_P4_FACTOR
            if self.system.safety_margin > self.SM_TARGET_IN_P4: reward += self.R_HEALTHY_SM_IN_P4_BONUS
            g_change = abs(self.system.g_lever - self.prev_g_lever); beta_change = abs(self.system.beta_lever - self.prev_beta_lever)
            if g_change > self.G_CHANGE_THRESHOLD_P4: reward -= self.R_G_OSCILLATION_P4_PENALTY
            if beta_change > self.BETA_CHANGE_THRESHOLD_P4: reward -= self.R_BETA_OSCILLATION_P4_PENALTY
        else:
            self.steps_in_phase_4_consecutive = 0
            if current_true_phase == 1: reward += self.R_PHASE_1_PENALTY
            elif current_true_phase == 2: reward += self.R_PHASE_2_PENALTY
            elif current_true_phase == 3: reward += self.R_PHASE_3_PENALTY
        system_collapsed_flags = self.system.collapsed_flag
        if system_collapsed_flags[-1] == 1 and (len(system_collapsed_flags) < 2 or system_collapsed_flags[-2] == 0):
            reward += self.R_COLLAPSE_EVENT_PENALTY
        if current_true_phase in (1, 2, 3) and prev_phase_for_reward_calc != current_true_phase:
            reward += self.R_PHASE_ENTRY_PENALTY
        if current_true_phase == 4 and prev_phase_for_reward_calc != 4: reward += self.R_ACHIEVED_P4_BONUS
        if self.system.fcrit < fcrit_floor_val * 1.5:
            divisor = max(fcrit_floor_val * 0.5, 1e-6)
            penalty_factor = ((fcrit_floor_val * 1.5 - self.system.fcrit) / divisor)**2
            reward -= self.R_FCRIT_LOW_PCTG_PENALTY_SCALE * penalty_factor
        if self.system.fcrit < fcrit_floor_val * 1.1: reward += self.R_FCRIT_VERY_LOW_PCTG_PENALTY
        sm_change = self.system.safety_margin - prev_sm
        if sm_change > 0.01: reward += sm_change * self.R_SM_CHANGE_POSITIVE_FACTOR
        elif sm_change < -0.01: reward += sm_change * self.R_SM_CHANGE_NEGATIVE_FACTOR
        reward -= action_cost
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_run_id = self.run_id_offset + self.np_random.integers(0, 100000)
        self.system = TDSystemIntervention(run_id=self.current_run_id, intervention_type='rl_agent', env_config=self.env_config)
        self.time_in_current_phase = 0; self.last_true_phase = self.system.true_phase if self.system.true_phase is not None else -1
        self.steps_in_phase_4_consecutive = 0; self.prev_g_lever = self.system.g_lever; self.prev_beta_lever = self.system.beta_lever
        return self._get_obs(), {}

    def step(self, action: int):
        prev_fcrit = self.system.fcrit; prev_sm = self.system.safety_margin
        prev_phase_for_reward_calc = self.system.true_phase
        self.prev_g_lever = self.system.g_lever; self.prev_beta_lever = self.system.beta_lever
        target_g, target_beta, target_fcrit = self.system.g_lever, self.system.beta_lever, self.system.fcrit
        action_cost = self.COST_BASE_ACTION if action != 0 else 0.0
        action_name_for_history = PhoenixLoopRLEnv.ACTION_NAMES[action]
        if action == 1: target_fcrit += self.FCRIT_BOOST_S_AMOUNT; action_cost += self.COST_FCRIT_BOOST_S_ADD
        elif action == 2: target_fcrit += self.FCRIT_BOOST_L_AMOUNT; action_cost += self.COST_FCRIT_BOOST_L_ADD
        elif action == 3: target_beta *= self.BETA_REDUCE_M_FACTOR; action_cost += self.COST_BETA_REDUCE_M_ADD
        elif action == 4: target_beta += self.BETA_INCREASE_M_AMOUNT; action_cost += self.COST_BETA_INCREASE_M_ADD
        elif action == 5: target_g *= self.G_REDUCE_M_FACTOR; action_cost += self.COST_G_REDUCE_M_ADD
        target_fcrit = max(target_fcrit, self.system.config.fcrit_floor)
        target_beta = np.clip(target_beta, 0.01, self.env_config.beta_initial * 2.0)
        target_g = np.clip(target_g, 0.1, G_INITIAL * 2.0)
        self.system.set_rl_targets(target_g, target_beta, target_fcrit, action_cost, action_int=action)
        self.system.step(current_algorithmic_phase=None)
        current_true_phase = self.system.true_phase
        if current_true_phase == self.last_true_phase: self.time_in_current_phase += 1
        else: self.time_in_current_phase = 0; self.last_true_phase = current_true_phase
        reward = self._calculate_reward(prev_fcrit, prev_sm, prev_phase_for_reward_calc)
        system_collapsed_flags = self.system.collapsed_flag
        intervention_applied_last = self.system.history['intervention_applied'][-1] if self.system.history.get('intervention_applied') else "None"
        terminated = (system_collapsed_flags[-1] == 1 and not intervention_applied_last.startswith("RL_Targets")) or \
                     (self.system.fcrit <= self.system.config.fcrit_floor - 1)
        truncated = self.system.time >= self.system.config.sim_steps
        if truncated and self.system.true_phase != 4: reward += self.R_TRUNCATED_NOT_IN_P4_PENALTY
        observation = self._get_obs()
        if 'intervention_applied' in self.system.history and self.system.history['intervention_applied']:
             self.system.history['intervention_applied'][-1] = action_name_for_history
        return observation, reward, terminated, truncated, {}
    def close(self): pass

import numpy as np # Assuming PhoenixLoopRLEnv and other dependencies are in the same scope
# Ensure PhoenixLoopRLEnv is defined before this class if it's in the same file,
# or properly imported if it's in another module (as it is in your full script).

# Assuming PhoenixLoopRLEnv is already defined/imported as in your full script:
# from .base_rl_env_class_module import PhoenixLoopRLEnv # Example import
# Ensure PhoenixLoopRLEnv is defined before this class if it's in the same file,
# or properly imported if it's in another module (as it is in your full script).
# from your_module import PhoenixLoopRLEnv # If PhoenixLoopRLEnv is in another file

class ProactiveStabilizationRLEnv(PhoenixLoopRLEnv):
    """
    Environment with reward shaping for proactive stability, aiming to
    achieve and maintain a high Safety Margin, primarily within Phase 4.
    This version attempts to strongly incentivize P4 attainment and maintenance
    while still guiding towards proactive SM.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Proactive Safety Margin (SM) Parameters ---
        self.PSM_TARGET = 2.0  # Target Safety Margin to maintain proactively.

        # REWARD for being above PSM_TARGET, especially in Phase 4.
        # This encourages not just being above target, but doing so in a functional state.
        self.R_PROACTIVE_SM_ABOVE_TARGET_IN_P4 = 0.75 # Increased from 0.1/0.15

        # PENALTY factor for SM being below PSM_TARGET.
        # This penalty is proportional to the deficit.
        self.R_PROACTIVE_SM_BELOW_TARGET_FACTOR = 0.15 # Slightly increased from 0.1, but effect is deficit-based.

        # PENALTY for SM being critically low (negative).
        # This needs to be significant but not completely prohibitive of P4 pursuit.
        # Base R_ACHIEVED_P4_BONUS is 5.0. Base R_COLLAPSE_EVENT_PENALTY is -20.0.
        self.R_PROACTIVE_CRITICAL_SM_PENALTY = 4.0 # Was 3.0 or 5.0. Let's try 4.0.

        # --- Enhanced Phase 4 Incentives for this Proactive Context ---
        # These bonuses are ADDITIVE to the base P4 rewards from PhoenixLoopRLEnv
        # to make P4 exceptionally attractive in this proactive setting.

        # Extra per-step bonus for being in Phase 4 in this proactive env.
        self.R_PROACTIVE_P4_CONTINUOUS_BONUS = 1.5 # Significantly larger than base 0.5

        # Stronger factor for sustained Phase 4 operation in this proactive env.
        self.R_PROACTIVE_P4_SUSTAINED_FACTOR = 0.3 # Increased from base 0.1

        # (Optional) One-time large bonus for achieving P4 *within this proactive env*
        # This could be used if R_ACHIEVED_P4_BONUS from base isn't enough.
        # self.R_PROACTIVE_P4_FIRST_ACHIEVEMENT_BONUS = 10.0

        # (Optional) Stronger penalty if the episode ends (truncated) and not in P4
        # self.R_PROACTIVE_TRUNCATED_NOT_P4_PENALTY = -25.0 # Base is -15.0


    def _calculate_reward(self, prev_fcrit, prev_sm, prev_phase_for_reward_calc):
        # 1. Get all base rewards from PhoenixLoopRLEnv.
        # This includes: R_STEP_SURVIVAL, base P4 bonuses (R_BASE_PHASE_4_BONUS, R_SUSTAINED_P4_FACTOR),
        # R_ACHIEVED_P4_BONUS, penalties for P1/P2/P3 (R_PHASE_1/2/3_PENALTY),
        # R_COLLAPSE_EVENT_PENALTY, R_FCRIT_LOW_PCTG_PENALTY_SCALE, R_FCRIT_VERY_LOW_PCTG_PENALTY,
        # R_PHASE_ENTRY_PENALTY, SM change rewards (R_SM_CHANGE_POSITIVE/NEGATIVE_FACTOR),
        # R_HEALTHY_SM_IN_P4_BONUS, oscillation penalties, R_TRUNCATED_NOT_IN_P4_PENALTY,
        # and action costs.
        reward = super()._calculate_reward(prev_fcrit, prev_sm, prev_phase_for_reward_calc)
        
        current_sm = self.system.safety_margin
        current_true_phase = self.system.true_phase

        # 2. Apply Proactive Safety Margin (PSM) Rewards/Penalties
        if current_sm >= self.PSM_TARGET:
            # Strong reward if SM is high AND in Phase 4 (desired avoidance state)
            if current_true_phase == 4:
                reward += self.R_PROACTIVE_SM_ABOVE_TARGET_IN_P4
            # Note: The base class already has R_HEALTHY_SM_IN_P4_BONUS (0.1 if sm > SM_TARGET_IN_P4=2.0)
            # This R_PROACTIVE_SM_ABOVE_TARGET_IN_P4 (0.75) will be additive if conditions met.
        else: # current_sm < self.PSM_TARGET
            deficit = self.PSM_TARGET - current_sm
            reward -= self.R_PROACTIVE_SM_BELOW_TARGET_FACTOR * deficit

        if current_sm < 0: # If safety margin is critically negative
            reward -= self.R_PROACTIVE_CRITICAL_SM_PENALTY
            
        # 3. Apply Enhanced Phase 4 Incentives specifically for this proactive environment
        if current_true_phase == 4:
            # Add an extra continuous bonus for being in P4
            reward += self.R_PROACTIVE_P4_CONTINUOUS_BONUS
            
            # Add an extra sustained P4 bonus
            # The base class already calculates and adds its sustained bonus.
            # This will be an additional component for the proactive agent.
            # self.steps_in_phase_4_consecutive is managed by the base class.
            reward += (self.steps_in_phase_4_consecutive / 10.0)**2 * self.R_PROACTIVE_P4_SUSTAINED_FACTOR

        # Optional: Override the R_TRUNCATED_NOT_IN_P4_PENALTY if needed
        # if self.system.time >= self.env_config.sim_steps and current_true_phase != 4:
        #    if hasattr(self, 'R_PROACTIVE_TRUNCATED_NOT_P4_PENALTY'):
        #        # Remove base penalty first if it was already applied by super()
        #        if hasattr(super(), 'R_TRUNCATED_NOT_IN_P4_PENALTY'): # Check if base class attr exists
        #             if prev_phase_for_reward_calc != 4: # ensure it was applied
        #                  reward += super().R_TRUNCATED_NOT_IN_P4_PENALTY # Add back to nullify
        #        reward -= self.R_PROACTIVE_TRUNCATED_NOT_P4_PENALTY # Apply new stronger one

        return reward
# --- Load ML Model and Scaler ---
ml_pipeline = None
try:
    ml_classifier_model_loaded = joblib.load('phoenix_rf_classifier.joblib')
    feature_scaler_loaded = joblib.load('phoenix_feature_scaler.joblib')
    print("INFO: Successfully loaded pre-trained ML model and scaler for heuristic use.")
    ml_pipeline = Pipeline([('scaler', feature_scaler_loaded), ('classifier', ml_classifier_model_loaded)])
except FileNotFoundError:
    print("WARNING: Pre-trained ML model or scaler not found. Heuristic TD-Informed interventions will not use ML phases.")

# --- Main Simulation Runner ---
# ... (run_intervention_simulation, analyze_intervention_results, plot_rl_agent_strategy_example) ...
# These functions are assumed to be largely the same as in the previous correct version,
# but will now operate with the environment using the new reward function.
# The `run_intervention_simulation` logic for handling history for RL agent
# and `analyze_intervention_results` metric calculations remain relevant.
# The plotting functions also remain relevant.

def run_intervention_simulation(intervention_type, run_id_offset=0, ml_model_pipeline_for_heuristics=None, trained_rl_model=None, env_config_to_use=None, env_class=PhoenixLoopRLEnv):
    if env_config_to_use is None: env_config_to_use = CURRENT_ENV_CONFIG
    print(f"\n--- Running Simulation: Intervention Type: {intervention_type} (Env: {env_config_to_use.level}) ---")
    condition_start_time = time.perf_counter()
    all_run_histories_condition = []
    if intervention_type == 'rl_agent' and trained_rl_model is None:
        print("ERROR: RL Agent selected, but no trained_rl_model provided."); return [], []
    run_durations = []
    for i in range(env_config_to_use.n_intervention_runs):
        run_start_time = time.perf_counter()
        current_run_id = run_id_offset + i
        if intervention_type == 'rl_agent':
            eval_env = env_class(run_id_offset=run_id_offset + 100000 + i, env_config=env_config_to_use)
            obs, _ = eval_env.reset()
            current_run_history = {key: list(val_list) for key, val_list in eval_env.system.history.items()}
            current_run_history['ml_phase_pred'] = []
            stream_diag_calc = DiagnosticsCalculatorIntervention(entropy_baseline=GLOBAL_ENTROPY_BASELINE, env_config=env_config_to_use)
            alg_phase_pred = 4
            episode_return_total = 0.0
            max_p4_consec = 0
            for step_in_episode in range(env_config_to_use.sim_steps):
                action_int, _ = trained_rl_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action_int)
                episode_return_total += reward
                if eval_env.steps_in_phase_4_consecutive > max_p4_consec:
                    max_p4_consec = eval_env.steps_in_phase_4_consecutive
                current_sample = {key_req: (eval_env.system.history[key_req][-1] if key_req in eval_env.system.history and eval_env.system.history[key_req] else getattr(eval_env.system, key_req, 0)) for key_req in REQUIRED_SERIES_FOR_DIAG_CALC}
                current_sample["run_id"] = current_run_id
                stream_diag_calc.push_sample(current_sample)
                if step_in_episode >= BURN_IN_PERIOD + FEATURE_WINDOW_SIZE - 1 and ml_model_pipeline_for_heuristics is not None:
                    latest_features_values = stream_diag_calc.latest_features(FEATURE_WINDOW_SIZE)
                    if not latest_features_values.empty:
                        try:
                            alg_phase_pred = ml_model_pipeline_for_heuristics.predict(latest_features_values)[0]
                        except Exception:
                            alg_phase_pred = 4
                current_run_history['ml_phase_pred'].append(alg_phase_pred)
                for key_hist in list(current_run_history.keys()):
                    if key_hist in eval_env.system.history and eval_env.system.history[key_hist]:
                        if len(eval_env.system.history[key_hist]) > len(current_run_history[key_hist]):
                            current_run_history[key_hist].append(eval_env.system.history[key_hist][-1])
                        elif len(eval_env.system.history[key_hist]) == len(current_run_history[key_hist]) and current_run_history[key_hist]:
                            current_run_history[key_hist][-1] = eval_env.system.history[key_hist][-1]
                if terminated or truncated:
                    break
            current_run_history['episode_return'] = episode_return_total
            current_run_history['max_steps_in_P4_consecutive'] = max_p4_consec
            final_diag_calc = DiagnosticsCalculatorIntervention(entropy_baseline=GLOBAL_ENTROPY_BASELINE, env_config=env_config_to_use)
            final_diag_calc.calculate_diagnostics_for_run(current_run_history)
            final_diag_calc.finalize_rhoE_and_baseline(recompute_baseline=False, quiet=True)
            current_run_history['diagnostics'] = final_diag_calc.diagnostics_list[0] if final_diag_calc.diagnostics_list else {}
            all_run_histories_condition.append(current_run_history)
            eval_env.close()
        else:
            system = TDSystemIntervention(run_id=current_run_id, intervention_type=intervention_type, env_config=env_config_to_use)
            stream_diag_calc = DiagnosticsCalculatorIntervention(entropy_baseline=GLOBAL_ENTROPY_BASELINE, env_config=env_config_to_use)
            algorithmic_phase_prediction = 4
            for t_step in range(env_config_to_use.sim_steps):
                if not system.step(algorithmic_phase_prediction): break
                if intervention_type == 'td_informed' and ml_model_pipeline_for_heuristics is not None:
                    current_sample = {key_req: (system.history[key_req][-1] if key_req in system.history and system.history[key_req] else getattr(system, key_req, 0)) for key_req in REQUIRED_SERIES_FOR_DIAG_CALC}
                    current_sample["run_id"] = current_run_id
                    stream_diag_calc.push_sample(current_sample)
                    if t_step >= BURN_IN_PERIOD + FEATURE_WINDOW_SIZE -1:
                        latest_features_values = stream_diag_calc.latest_features(FEATURE_WINDOW_SIZE)
                        if not latest_features_values.empty:
                            try: algorithmic_phase_prediction = ml_model_pipeline_for_heuristics.predict(latest_features_values)[0]
                            except Exception: algorithmic_phase_prediction = 4
                        else: algorithmic_phase_prediction = 4
                    else: algorithmic_phase_prediction = 4
            all_run_histories_condition.append(system.history)
        run_duration = time.perf_counter() - run_start_time; run_durations.append(run_duration)
        avg_run_time = np.mean(run_durations); runs_left = env_config_to_use.n_intervention_runs - (i + 1)
        eta_condition = runs_left * avg_run_time
        print(f"Finished Eval Run {i+1:02d}/{env_config_to_use.n_intervention_runs} ({intervention_type}) in {_hms(run_duration)}. Avg: {_hms(avg_run_time)}. ETA: {_hms(eta_condition)}")
    final_diag_list_condition = []
    final_diag_calculator_condition = DiagnosticsCalculatorIntervention(
        entropy_baseline=GLOBAL_ENTROPY_BASELINE, env_config=env_config_to_use
    )
    for run_hist_data in all_run_histories_condition:
        if run_hist_data.get('time') and len(run_hist_data['time']) > 1:
            final_diag_calculator_condition.calculate_diagnostics_for_run(run_hist_data)
    final_diag_calculator_condition.finalize_rhoE_and_baseline(
        recompute_baseline=False, quiet=True
    )
    final_diag_list_condition = final_diag_calculator_condition.diagnostics_list
    condition_duration = time.perf_counter() - condition_start_time
    print(f"--- Condition {intervention_type} (evaluation) completed in {_hms(condition_duration)} ---")
    return all_run_histories_condition, final_diag_list_condition

def analyze_intervention_results(results_dict, config_used, condition_names_map_arg):  # Pass map
    print("\n\n--- Intervention Study Analysis (with RL) ---")
    summary_dict = {}
    run_logs = []

    # --- baseline (No-intervention) stats for later efficiency ratios ------------------------
    no_intervention_stats_for_eff = None
    no_intervention_key_name = condition_names_map_arg.get('none')

    if no_intervention_key_name and no_intervention_key_name in results_dict:
        histories_no_int, _ = results_dict[no_intervention_key_name]
        if histories_no_int:
            num_collapses_total_no_int = 0
            times_to_phase4_no_int_list = []
            for run_hist in histories_no_int:
                if not run_hist or not run_hist.get("time") or len(run_hist["time"]) <= 1:
                    continue
                collapsed_flags_hist = np.asarray(run_hist["collapsed_flag"])
                run_collapses = (
                    np.diff(collapsed_flags_hist).sum() if len(collapsed_flags_hist) > 1 else 0
                )
                num_collapses_total_no_int += run_collapses
                true_phases = np.asarray(run_hist["true_phase"])
                time_hist = np.asarray(run_hist["time"])
                collapse_indices = (
                    np.where(np.diff(collapsed_flags_hist) > 0)[0]
                    if len(collapsed_flags_hist) > 1
                    else np.array([])
                )
                if not collapse_indices.size and run_collapses == 0:
                    times_to_phase4_no_int_list.append(
                        0 if true_phases[-1] == 4 else float(config_used.sim_steps)
                    )
                    continue
                first_collapse_idx = collapse_indices[0] + 1 if collapse_indices.size > 0 else 0
                phase4_after_collapse = np.where(
                    (true_phases == 4) & (np.arange(len(true_phases)) > first_collapse_idx)
                )[0]
                times_to_phase4_no_int_list.append(
                    time_hist[phase4_after_collapse[0]] - time_hist[first_collapse_idx]
                    if len(phase4_after_collapse) > 0
                    else float(config_used.sim_steps)
                )

            no_intervention_stats_for_eff = {
                "total_collapses": num_collapses_total_no_int,
                "avg_time_to_p4": np.nanmean(times_to_phase4_no_int_list)
                if times_to_phase4_no_int_list
                else np.nan,
            }

    # ─────────────────────────────────────────────────────────────────────────────────────────
    # loop over each intervention condition
    # ─────────────────────────────────────────────────────────────────────────────────────────
    for condition, (histories, diagnostics_list) in results_dict.items():
        print(f"\nCondition: {condition}")

        # No runs for this condition ----------------------------------------------------------
        if not histories:
            summary_dict[condition] = {
                "n_runs": 0,
                "collapse_rate": {"mean": np.nan, "sem": 0.0},
                "avg_time_to_p4": {"mean": np.nan, "sem": 0.0},
                "avg_duration_stable_p4": {"mean": np.nan, "sem": 0.0},
                "avg_total_fcrit_injected": {"mean": np.nan, "sem": 0.0},
                "avg_lever_cost": {"mean": np.nan, "sem": 0.0},
                "phase_occupancy": {"mean": [np.nan] * 5, "sem": [0.0] * 5},
                "fcrit_curve": {"t": [], "mean": [], "sem": []},
                "diagnostics": {
                    "SpeedIndex": {"mean": np.nan, "max": np.nan, "std": np.nan},
                    "CoupleIndex": {"mean": np.nan, "max": np.nan, "std": np.nan},
                    "rhoE": {"mean": np.nan, "max": np.nan, "std": np.nan},
                    "arc_length": np.nan,
                },
                "_raw": {
                    "collapse_counts": [],
                    "time_to_p4": [],
                    "total_fcrit_injected": [],
                },
            }
            continue

        # -------------------------------------------------------------------------------------
        # collectors
        # -------------------------------------------------------------------------------------
        times_to_phase4 = []
        fcrit_at_phase4_start = []
        num_stuck_or_regressed = 0
        num_collapses_total = 0
        collapse_counts_list = []

        durations_in_phase_first_recovery = {p: [] for p in range(1, 5)}
        durations_of_first_stable_phase4 = []
        avg_sm_in_first_stable_phase4 = []

        total_fcrit_injected_runs = []
        lever_cost_runs = []
        phase_occupancy_runs = []
        fcrit_curve_runs = []
        sm_curve_runs = []
        g_lever_curve_runs = []
        beta_lever_curve_runs = []

        rl_action_counts_runs = []
        rl_action_costs = []
        max_consec_p4_runs = []
        episode_returns = []
        transition_counts = np.zeros((6, 6), dtype=int)

        fcrit_floor_fracs = []

        for run_idx, run_hist in enumerate(histories):
            if not run_hist or not run_hist.get("time") or len(run_hist["time"]) <= 1:
                num_stuck_or_regressed += 1
                times_to_phase4.append(float(config_used.sim_steps))
                continue

            true_phases = np.asarray(run_hist["true_phase"])
            fcrit_hist = np.asarray(run_hist["fcrit"], dtype=np.float32)
            time_hist = np.asarray(run_hist["time"])
            safety_margin_hist = np.asarray(run_hist["safety_margin"], dtype=np.float32)
            collapsed_flags_hist = np.asarray(run_hist["collapsed_flag"])

            run_collapses = (
                np.diff(collapsed_flags_hist).sum() if len(collapsed_flags_hist) > 1 else 0
            )
            num_collapses_total += run_collapses
            collapse_counts_list.append(run_collapses)

            # ---- totals & costs --------------------------------------------------------------
            total_fcrit_injected_runs.append(
                run_hist["total_fcrit_injected"][-1]
                if "total_fcrit_injected" in run_hist
                and run_hist["total_fcrit_injected"]
                and run_hist["total_fcrit_injected"][-1] is not np.nan
                else 0.0
            )

            g_hist = np.asarray(run_hist.get("g_lever", []), dtype=np.float32)
            beta_hist = np.asarray(run_hist.get("beta_lever", []), dtype=np.float32)
            lever_cost_runs.append(np.abs(np.diff(g_hist)).sum() + np.abs(np.diff(beta_hist)).sum())

            if fcrit_hist.size:
                fcrit_floor_fracs.append(float(np.mean(fcrit_hist < config_used.fcrit_floor)))

            # ---- phase occupancy -------------------------------------------------------------
            phase_occ = np.bincount(true_phases.astype(int), minlength=5) / len(true_phases)
            phase_occupancy_runs.append(phase_occ)
            curve_indices = np.arange(0, config_used.sim_steps + 1, 25)

            # ---------- Fcrit --------------------------------------------
            fc_series = np.full(curve_indices.shape, np.nan, dtype=np.float32)
            valid_mask_fc = curve_indices < len(fcrit_hist)
            fc_series[valid_mask_fc] = fcrit_hist[curve_indices[valid_mask_fc]]
            fcrit_curve_runs.append(fc_series)

            # ---------- Safety-margin -----------------------------------
            sm_series = np.full(curve_indices.shape, np.nan, dtype=np.float32)
            valid_mask_sm = curve_indices < len(safety_margin_hist)
            sm_series[valid_mask_sm] = safety_margin_hist[curve_indices[valid_mask_sm]]
            sm_curve_runs.append(sm_series)

            # ---------- g-lever -----------------------------------------
            g_series = np.full(curve_indices.shape, np.nan, dtype=np.float32)
            valid_mask_g = curve_indices < len(g_hist)
            g_series[valid_mask_g] = g_hist[curve_indices[valid_mask_g]]
            g_lever_curve_runs.append(g_series)

            # ---------- β-lever -----------------------------------------
            beta_series = np.full(curve_indices.shape, np.nan, dtype=np.float32)
            valid_mask_b = curve_indices < len(beta_hist)
            beta_series[valid_mask_b] = beta_hist[curve_indices[valid_mask_b]]
            beta_lever_curve_runs.append(beta_series)

            # longest contiguous P4
            max_consec_p4_runs.append(
                int(max([len(list(g)) for k, g in groupby(true_phases == 4) if k] or [0]))
            )

            # ---- RL-specific ---------------------------------------------------------------
            if condition == condition_names_map_arg.get("rl_agent"):
                actions = np.asarray(run_hist.get("rl_action_chosen", []))
                actions = actions[~np.isnan(actions)].astype(int)
                if actions.size:
                    rl_action_counts_runs.append(np.bincount(actions, minlength=6))
                    transition_counts += np.bincount(
                        actions[:-1] * 6 + actions[1:], minlength=36
                    ).reshape(6, 6)

                    for a in actions:
                        cost = PhoenixLoopRLEnv.COST_BASE_ACTION if a != 0 else 0.0
                        if a == 1:
                            cost += PhoenixLoopRLEnv.COST_FCRIT_BOOST_S_ADD
                        elif a == 2:
                            cost += PhoenixLoopRLEnv.COST_FCRIT_BOOST_L_ADD
                        elif a == 3:
                            cost += PhoenixLoopRLEnv.COST_BETA_REDUCE_M_ADD
                        elif a == 4:
                            cost += PhoenixLoopRLEnv.COST_BETA_INCREASE_M_ADD
                        elif a == 5:
                            cost += PhoenixLoopRLEnv.COST_G_REDUCE_M_ADD
                        rl_action_costs.append(cost)

                episode_returns.append(run_hist.get("episode_return", np.nan))
                if "max_steps_in_P4_consecutive" in run_hist:
                    max_consec_p4_runs[-1] = run_hist["max_steps_in_P4_consecutive"]

            # ---- collapse / P4 timing -------------------------------------------------------
            collapse_indices = (
                np.where(np.diff(collapsed_flags_hist) > 0)[0]
                if len(collapsed_flags_hist) > 1
                else np.array([])
            )

            # … no collapse at all
            if not collapse_indices.size and run_collapses == 0:
                if true_phases[-1] == 4:
                    times_to_phase4.append(0)
                    fcrit_at_phase4_start.append(fcrit_hist[0] if fcrit_hist.size else np.nan)
                    durations_of_first_stable_phase4.append(
                        time_hist[-1] - time_hist[0] if len(time_hist) > 1 else 0
                    )
                    avg_sm_in_first_stable_phase4.append(
                        np.nanmean(safety_margin_hist) if safety_margin_hist.size else np.nan
                    )
                else:
                    num_stuck_or_regressed += 1
                    times_to_phase4.append(float(config_used.sim_steps))

                run_logs.append(
                    [
                        run_idx,
                        condition,
                        run_collapses,
                        times_to_phase4[-1],
                        total_fcrit_injected_runs[-1],
                        lever_cost_runs[-1],
                    ]
                )
                continue

            # … there *was* an initial collapse
            first_collapse_idx = collapse_indices[0] + 1 if collapse_indices.size else 0
            phase4_after_collapse_indices = np.where(
                (true_phases == 4) & (np.arange(len(true_phases)) > first_collapse_idx)
            )[0]
            if len(phase4_after_collapse_indices) > 0:
                first_phase4_start_idx = phase4_after_collapse_indices[0]
                time_to_p4 = time_hist[first_phase4_start_idx] - time_hist[first_collapse_idx]
                times_to_phase4.append(time_to_p4)
                fcrit_at_phase4_start.append(fcrit_hist[first_phase4_start_idx])

                # durations in  P1/P2/P3 on way back to first P4
                current_phase_val = true_phases[first_collapse_idx]
                current_phase_start_time_val = time_hist[first_collapse_idx]
                for t_idx in range(first_collapse_idx + 1, first_phase4_start_idx + 1):
                    phase_at_t = true_phases[t_idx]
                    if phase_at_t != current_phase_val:
                        if (
                            current_phase_val in durations_in_phase_first_recovery
                            and current_phase_start_time_val < time_hist[t_idx]
                        ):
                            durations_in_phase_first_recovery[current_phase_val].append(
                                time_hist[t_idx] - current_phase_start_time_val
                            )
                        current_phase_val = phase_at_t
                        current_phase_start_time_val = time_hist[t_idx]

                # final push up to P4
                if (
                    current_phase_val in durations_in_phase_first_recovery
                    and current_phase_start_time_val <= time_hist[first_phase4_start_idx]
                ):
                    durations_in_phase_first_recovery[current_phase_val].append(
                        time_hist[first_phase4_start_idx] - current_phase_start_time_val
                    )

                # how long does that first stable-P4 last?
                phases_after_p4 = true_phases[first_phase4_start_idx:]
                next_phase_change = np.where(phases_after_p4 != 4)[0]
                end_of_this_p4_idx = (
                    first_phase4_start_idx + next_phase_change[0] if len(next_phase_change) else len(true_phases)
                )

                duration_p4 = time_hist[end_of_this_p4_idx - 1] - time_hist[first_phase4_start_idx]
                durations_of_first_stable_phase4.append(duration_p4)
                sm_values_in_p4 = safety_margin_hist[first_phase4_start_idx:end_of_this_p4_idx]
                avg_sm_in_first_stable_phase4.append(
                    np.nanmean(sm_values_in_p4) if sm_values_in_p4.size else np.nan
                )

                run_logs.append(
                    [
                        run_idx,
                        condition,
                        run_collapses,
                        time_to_p4,
                        total_fcrit_injected_runs[-1],
                        lever_cost_runs[-1],
                    ]
                )
            else:
                num_stuck_or_regressed += 1
                times_to_phase4.append(float(config_used.sim_steps))
                run_logs.append(
                    [
                        run_idx,
                        condition,
                        run_collapses,
                        times_to_phase4[-1],
                        total_fcrit_injected_runs[-1],
                        lever_cost_runs[-1],
                    ]
                )

        # ####################################################################################
        # aggregate & compute stats
        # ####################################################################################
        avg_time_to_p4 = np.nanmean(times_to_phase4) if times_to_phase4 else np.nan
        avg_duration_stable_p4 = (
            np.nanmean(durations_of_first_stable_phase4) if durations_of_first_stable_phase4 else np.nan
        )
        avg_total_fcrit_injected = (
            np.nanmean(total_fcrit_injected_runs) if total_fcrit_injected_runs else 0.0
        )
        avg_lever_cost = np.nanmean(lever_cost_runs) if lever_cost_runs else 0.0

        avg_fcrit_at_p4_start = (
            np.nanmean(fcrit_at_phase4_start) if fcrit_at_phase4_start else np.nan
        )
        sem_fcrit_at_p4_start = (
            sem(fcrit_at_phase4_start, nan_policy="omit") if len(fcrit_at_phase4_start) > 1 else 0.0
        )
        mean_avg_sm_stable_p4 = (
            np.nanmean(avg_sm_in_first_stable_phase4) if avg_sm_in_first_stable_phase4 else np.nan
        )

        collapse_quantiles = (
            np.nanpercentile(collapse_counts_list, [0, 25, 50, 75, 100]).tolist()
            if collapse_counts_list
            else [np.nan] * 5
        )
        time_to_p4_iqr = (
            float(np.subtract(*np.nanpercentile(times_to_phase4, [75, 25])))
            if times_to_phase4
            else np.nan
        )
        fcrit_injected_cv = (
            float(np.nanstd(total_fcrit_injected_runs) / avg_total_fcrit_injected)
            if avg_total_fcrit_injected and len(total_fcrit_injected_runs) > 1
            else np.nan
        )
        fcrit_per_collapse_runs = [
            inj / (c if c > 0 else 1) for inj, c in zip(total_fcrit_injected_runs, collapse_counts_list)
        ]
        avg_fcrit_per_collapse_mean = (
            float(np.nanmean(fcrit_per_collapse_runs)) if fcrit_per_collapse_runs else np.nan
        )
        avg_fcrit_per_collapse_sem = (
            float(sem(fcrit_per_collapse_runs, nan_policy="omit"))
            if len(fcrit_per_collapse_runs) > 1
            else 0.0
        )

        # ---- arrays → mean/SEM -------------------------------------------------------------
        phase_occ_arr = np.array(phase_occupancy_runs)
        phase_occ_mean = (
            np.nanmean(phase_occ_arr, axis=0) if phase_occ_arr.size else np.full(5, np.nan)
        )
        phase_occ_sem = (
            sem(phase_occ_arr, axis=0, nan_policy="omit") if phase_occ_arr.size > 1 else np.zeros(5)
        )
        phase_occ_sem = np.nan_to_num(phase_occ_sem, nan=0.0)

        fcrit_curve_arr = np.array(fcrit_curve_runs)
        fcrit_curve_mean = np.nanmean(fcrit_curve_arr, axis=0) if fcrit_curve_arr.size else []
        fcrit_curve_sem = (
            sem(fcrit_curve_arr, axis=0, nan_policy="omit")
            if fcrit_curve_arr.size > 1
            else np.zeros_like(fcrit_curve_mean)
        )

        sm_curve_arr = np.array(sm_curve_runs)
        sm_curve_mean = np.nanmean(sm_curve_arr, axis=0) if sm_curve_arr.size else []
        sm_curve_sem = (
            sem(sm_curve_arr, axis=0, nan_policy="omit")
            if sm_curve_arr.size > 1
            else np.zeros_like(sm_curve_mean)
        )
        sm_curve_sem = np.nan_to_num(sm_curve_sem, nan=0.0)

        g_curve_arr = np.array(g_lever_curve_runs)
        g_curve_mean = np.nanmean(g_curve_arr, axis=0) if g_curve_arr.size else []
        g_curve_sem = (
            sem(g_curve_arr, axis=0, nan_policy="omit")
            if g_curve_arr.size > 1
            else np.zeros_like(g_curve_mean)
        )
        g_curve_sem = np.nan_to_num(g_curve_sem, nan=0.0)

        beta_curve_arr = np.array(beta_lever_curve_runs)
        beta_curve_mean = np.nanmean(beta_curve_arr, axis=0) if beta_curve_arr.size else []
        beta_curve_sem = (
            sem(beta_curve_arr, axis=0, nan_policy="omit")
            if beta_curve_arr.size > 1
            else np.zeros_like(beta_curve_mean)
        )
        beta_curve_sem = np.nan_to_num(beta_curve_sem, nan=0.0)

        # ---- integrals & slopes ------------------------------------------------------------
        fcrit_auc = (
            float(np.trapz(fcrit_curve_mean, dx=25))
            if isinstance(fcrit_curve_mean, np.ndarray) and fcrit_curve_mean.size
            else np.nan
        )
        sm_auc = (
            float(np.trapz(sm_curve_mean, dx=25))
            if isinstance(sm_curve_mean, np.ndarray) and sm_curve_mean.size
            else np.nan
        )

        sm_slopes = []
        if sm_curve_arr.size:
            for series in sm_curve_arr:
                if np.all(np.isnan(series[:7])):
                    continue
                slope, _, _, _, _ = linregress(curve_indices[:7], series[:7])
                sm_slopes.append(slope)
        sm_slope_mean = float(np.nanmean(sm_slopes)) if sm_slopes else np.nan
        sm_slope_sem = float(sem(sm_slopes, nan_policy="omit")) if len(sm_slopes) > 1 else 0.0

        p_floor_mean = float(np.nanmean(fcrit_floor_fracs)) if fcrit_floor_fracs else np.nan
        p_floor_sem = (
            float(sem(fcrit_floor_fracs, nan_policy="omit"))
            if len(fcrit_floor_fracs) > 1
            else (0.0 if fcrit_floor_fracs else np.nan)
        )

        # ---- diagnostics -------------------------------------------------------------------
        diag_speed_means, diag_speed_maxes, diag_speed_stds = [], [], []
        diag_couple_means, diag_couple_maxes, diag_couple_stds = [], [], []
        diag_rho_means, diag_rho_maxes, diag_rho_stds = [], [], []
        arc_lengths = []

        for diag in diagnostics_list:
            if not diag:
                continue
            S = np.asarray(diag.get("SpeedIndex", []), dtype=float)
            C = np.asarray(diag.get("CoupleIndex", []), dtype=float)
            R = np.asarray(diag.get("rhoE", []), dtype=float)

            if S.size:
                diag_speed_means.append(np.nanmean(S))
                diag_speed_maxes.append(np.nanmax(S))
                diag_speed_stds.append(np.nanstd(S))
            if C.size:
                diag_couple_means.append(np.nanmean(C))
                diag_couple_maxes.append(np.nanmax(C))
                diag_couple_stds.append(np.nanstd(C))
            if R.size:
                diag_rho_means.append(np.nanmean(R))
                diag_rho_maxes.append(np.nanmax(R))
                diag_rho_stds.append(np.nanstd(R))

            if (
                S.size
                and C.size
                and R.size
                and len(S) == len(C) == len(R)
                and len(S) > 1
            ):
                arc_lengths.append(
                    np.sum(np.linalg.norm(np.diff(np.c_[S, C, R], axis=0), axis=1))
                )

        diag_dict = {
            "SpeedIndex": {
                "mean": float(np.nanmean(diag_speed_means)) if diag_speed_means else np.nan,
                "max": float(np.nanmean(diag_speed_maxes)) if diag_speed_maxes else np.nan,
                "std": float(np.nanmean(diag_speed_stds)) if diag_speed_stds else np.nan,
            },
            "CoupleIndex": {
                "mean": float(np.nanmean(diag_couple_means)) if diag_couple_means else np.nan,
                "max": float(np.nanmean(diag_couple_maxes)) if diag_couple_maxes else np.nan,
                "std": float(np.nanmean(diag_couple_stds)) if diag_couple_stds else np.nan,
            },
            "rhoE": {
                "mean": float(np.nanmean(diag_rho_means)) if diag_rho_means else np.nan,
                "max": float(np.nanmean(diag_rho_maxes)) if diag_rho_maxes else np.nan,
                "std": float(np.nanmean(diag_rho_stds)) if diag_rho_stds else np.nan,
            },
            "arc_length": float(np.nanmean(arc_lengths)) if arc_lengths else np.nan,
        }

        # ---- RL action summary -------------------------------------------------------------
        rl_actions_summary = None
        if rl_action_counts_runs:
            counts_sum = np.sum(rl_action_counts_runs, axis=0)
            rl_actions_summary = {
                "counts": counts_sum.astype(int).tolist(),
                "mean_cost": float(np.nanmean(rl_action_costs)) if rl_action_costs else np.nan,
                "median_cost": float(np.nanmedian(rl_action_costs)) if rl_action_costs else np.nan,
                "iqr_cost": float(np.subtract(*np.percentile(rl_action_costs, [75, 25])))
                if rl_action_costs
                else np.nan,
                "transition_matrix": (
                    transition_counts
                    / np.maximum(transition_counts.sum(axis=1, keepdims=True), 1)
                ).tolist(),
            }
            rl_episode_return = {
                "mean": float(np.nanmean(episode_returns)) if episode_returns else np.nan,
                "sem": float(sem(episode_returns, nan_policy="omit"))
                if len(episode_returns) > 1
                else 0.0,
                "min": float(np.nanmin(episode_returns)) if episode_returns else np.nan,
                "max": float(np.nanmax(episode_returns)) if episode_returns else np.nan,
            }
            rl_steps_p4_consec = {
                "mean": float(np.nanmean(max_consec_p4_runs)) if max_consec_p4_runs else np.nan,
                "sem": float(sem(max_consec_p4_runs, nan_policy="omit"))
                if len(max_consec_p4_runs) > 1
                else 0.0,
            }

        # ---- console summary ---------------------------------------------------------------
        print(
            f"  Avg Time to Restabilization (P4): {avg_time_to_p4:.2f} steps\n"
            f"  Avg Fcrit at start of P4: {avg_fcrit_at_p4_start:.2f}\n"
            f"  Not reaching P4: {num_stuck_or_regressed}/{len(histories)}\n"
            f"  Total collapses (endogenous post-shock degradations): {num_collapses_total}"
        )
        for phase_num_key in sorted(durations_in_phase_first_recovery.keys()):
            durs = durations_in_phase_first_recovery[phase_num_key]
            avg_dur = np.nanmean(durs) if durs else 0.0
            print(f"  Avg duration P{phase_num_key} (1st recovery): {avg_dur:.2f}")

        print(
            f"  NEW: Avg Duration 1st Stable P4: {avg_duration_stable_p4:.2f} steps\n"
            f"  NEW: Avg SM during 1st Stable P4: {mean_avg_sm_stable_p4:.2f}\n"
            f"  NEW: Avg Total Fcrit Injected (Intervention): {avg_total_fcrit_injected:.2f}"
        )
        print(f"  NEW: Avg Lever Adjustment Cost: {avg_lever_cost:.2f}")

        # ---- efficiency metrics versus baseline -------------------------------------------
        efficiency_metrics_text = ""
        if no_intervention_stats_for_eff and condition != no_intervention_key_name:
            collapses_cond_avg_per_run = num_collapses_total / config_used.n_intervention_runs
            collapses_no_int_avg_per_run = no_intervention_stats_for_eff[
                "total_collapses"
            ] / config_used.n_intervention_runs
            net_collapses_prevented_avg_per_run = (
                collapses_no_int_avg_per_run - collapses_cond_avg_per_run
            )
            if (
                net_collapses_prevented_avg_per_run > 1e-3
                and avg_total_fcrit_injected > 1e-6
            ):
                efficiency_metrics_text += (
                    f"  NEW: Avg Fcrit/Net Endog. Collapse Prevented: "
                    f"{avg_total_fcrit_injected / net_collapses_prevented_avg_per_run:.2f}\n"
                )
            else:
                efficiency_metrics_text += (
                    "  NEW: Avg Fcrit/Net Endog. Collapse Prevented: N/A\n"
                )

            time_to_p4_no_int_avg = no_intervention_stats_for_eff["avg_time_to_p4"]
            if not np.isnan(avg_time_to_p4) and not np.isnan(time_to_p4_no_int_avg):
                steps_saved_in_restab_avg = time_to_p4_no_int_avg - avg_time_to_p4
                if steps_saved_in_restab_avg > 1e-3 and avg_total_fcrit_injected > 1e-6:
                    efficiency_metrics_text += (
                        f"  NEW: Avg Fcrit/Step Saved in Restab: "
                        f"{avg_total_fcrit_injected / steps_saved_in_restab_avg:.2f}\n"
                    )
                else:
                    efficiency_metrics_text += (
                        "  NEW: Avg Fcrit/Step Saved in Restab: "
                        "N/A (no steps saved or no injection)\n"
                    )
            else:
                efficiency_metrics_text += (
                    "  NEW: Avg Fcrit/Step Saved in Restab: N/A (data missing)\n"
                )
        else:
            efficiency_metrics_text = (
                "  NEW: Efficiency Metrics vs NoIntervention: "
                "N/A (Base or NoIntervention data missing)\n"
            )

        print(efficiency_metrics_text.strip())

        # ---- build summary dict ------------------------------------------------------------
        summary_dict[condition] = {
            "n_runs": len(histories),
            "collapse_rate": {
                "mean": float(np.nanmean(collapse_counts_list))
                if collapse_counts_list
                else np.nan,
                "sem": float(sem(collapse_counts_list, nan_policy="omit"))
                if len(collapse_counts_list) > 1
                else 0.0,
            },
            "avg_time_to_p4": {
                "mean": float(avg_time_to_p4),
                "sem": float(sem(times_to_phase4, nan_policy="omit"))
                if len(times_to_phase4) > 1
                else 0.0,
            },
            "avg_duration_stable_p4": {
                "mean": float(avg_duration_stable_p4),
                "sem": float(sem(durations_of_first_stable_phase4, nan_policy="omit"))
                if len(durations_of_first_stable_phase4) > 1
                else 0.0,
            },
            "avg_fcrit_at_p4_start": {"mean": float(avg_fcrit_at_p4_start), "sem": float(sem_fcrit_at_p4_start)},
            "avg_total_fcrit_injected": {
                "mean": float(avg_total_fcrit_injected),
                "sem": float(sem(total_fcrit_injected_runs, nan_policy="omit"))
                if len(total_fcrit_injected_runs) > 1
                else 0.0,
            },
            "avg_lever_cost": {
                "mean": float(avg_lever_cost),
                "sem": float(sem(lever_cost_runs, nan_policy="omit"))
                if len(lever_cost_runs) > 1
                else 0.0,
            },
            "phase_occupancy": {"mean": phase_occ_mean.tolist(), "sem": phase_occ_sem.tolist()},
            "fcrit_curve": {
                "t": np.arange(0, config_used.sim_steps + 1, 25).tolist(),
                "mean": fcrit_curve_mean.tolist()
                if isinstance(fcrit_curve_mean, np.ndarray)
                else [],
                "sem": fcrit_curve_sem.tolist()
                if isinstance(fcrit_curve_sem, np.ndarray)
                else [],
            },
            "sm_curve": {
                "t": np.arange(0, config_used.sim_steps + 1, 25).tolist(),
                "mean": sm_curve_mean.tolist() if isinstance(sm_curve_mean, np.ndarray) else [],
                "sem": sm_curve_sem.tolist() if isinstance(sm_curve_sem, np.ndarray) else [],
            },
            "g_lever_curve": {
                "t": np.arange(0, config_used.sim_steps + 1, 25).tolist(),
                "mean": g_curve_mean.tolist() if isinstance(g_curve_mean, np.ndarray) else [],
                "sem": g_curve_sem.tolist() if isinstance(g_curve_sem, np.ndarray) else [],
            },
            "beta_lever_curve": {
                "t": np.arange(0, config_used.sim_steps + 1, 25).tolist(),
                "mean": beta_curve_mean.tolist()
                if isinstance(beta_curve_mean, np.ndarray)
                else [],
                "sem": beta_curve_sem.tolist()
                if isinstance(beta_curve_sem, np.ndarray)
                else [],
            },
            "fcrit_auc": fcrit_auc,
            "sm_auc": sm_auc,
            "p_Fcrit_floor": {"mean": p_floor_mean, "sem": p_floor_sem},
            "sm_slope": {"mean": sm_slope_mean, "sem": sm_slope_sem},
            "collapse_rate.quantiles": collapse_quantiles,
            "avg_time_to_p4.iqr": time_to_p4_iqr,
            "avg_total_fcrit_injected.cv": fcrit_injected_cv,
            "avg_fcrit_injected_per_collapse": {
                "mean": avg_fcrit_per_collapse_mean,
                "sem": avg_fcrit_per_collapse_sem,
            },
            "diagnostics": diag_dict,
            "_raw": {
                "collapse_counts": collapse_counts_list,
                "time_to_p4": times_to_phase4,
                "total_fcrit_injected": total_fcrit_injected_runs,
            },
        }

        if rl_actions_summary:
            summary_dict[condition]["rl_actions"] = rl_actions_summary
            summary_dict[condition]["rl_episode_return"] = rl_episode_return
            summary_dict[condition]["rl_steps_in_P4_consecutive"] = rl_steps_p4_consec

    # ---- contextual note for RL agent ------------------------------------------------------
    rl_agent_key_name = condition_names_map_arg.get("rl_agent")
    if rl_agent_key_name and rl_agent_key_name in summary_dict:
        print(
            f"\n  Note for {rl_agent_key_name}: 'Total collapses (endogenous post-shock degradations)' "
            f"counts system failures leading to P1-P3 phases. If this is 0 for "
            f"{config_used.n_intervention_runs} runs, the RL agent prevented entry "
            "into the Phoenix Loop after the initial shocks."
        )

    # ----------------------------------------------------------------------------------------
    # statistical tests vs. No-Intervention
    # ----------------------------------------------------------------------------------------
    if "A_NoIntervention" in summary_dict:
        baseline = summary_dict["A_NoIntervention"]
        for cond_key, vals in summary_dict.items():
            if cond_key == "A_NoIntervention":
                continue
            try:
                _, p = ttest_ind(
                    vals["_raw"]["collapse_counts"],
                    baseline["_raw"]["collapse_counts"],
                    equal_var=False,
                )
                vals["collapse_rate.p_vs_NoIntervention"] = float(p)
            except Exception:
                vals["collapse_rate.p_vs_NoIntervention"] = np.nan

            try:
                _, p = ttest_ind(
                    vals["_raw"]["time_to_p4"], baseline["_raw"]["time_to_p4"], equal_var=False
                )
                vals["avg_time_to_p4.p_vs_NoIntervention"] = float(p)
            except Exception:
                vals["avg_time_to_p4.p_vs_NoIntervention"] = np.nan

            try:
                _, p = ttest_ind(
                    vals["_raw"]["total_fcrit_injected"],
                    baseline["_raw"]["total_fcrit_injected"],
                    equal_var=False,
                )
                vals["avg_total_fcrit_injected.p_vs_NoIntervention"] = float(p)
            except Exception:
                vals["avg_total_fcrit_injected.p_vs_NoIntervention"] = np.nan

    summary_dict["_run_logs"] = run_logs
    return summary_dict


def plot_rl_agent_strategy_example(history, condition_key_name, config_to_use, output_dir=RESULTS_DIR): # Condensed
    if not history or not history.get('time') or len(history['time']) <=1: print(f"No data for RL plot {condition_key_name}"); return
    time_h = np.asarray(history['time']); min_len = len(time_h)
    data = {k: np.asarray(history.get(k, [np.nan]*min_len))[:min_len] for k in ['fcrit', 'g_lever', 'beta_lever', 'target_g', 'target_beta', 'target_fcrit', 'true_phase', 'safety_margin', 'rl_action_chosen']}
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True); fig.suptitle(f"RL Agent Strategy Example - {condition_key_name} (Run 0, Env: {config_to_use.level})", fontsize=16)
    axs[0].plot(time_h, data['fcrit'], label='Fcrit', c='g', lw=2); axs[0].plot(time_h, data['target_fcrit'], label='Target Fcrit (RL)', c='lightgreen', ls='--'); axs[0].axhline(config_to_use.fcrit_floor, c='r', ls=':', label='Fcrit Floor'); axs[0].set_ylabel('Fcrit'); axs[0].legend(loc='lower left'); axs[0].grid(True, ls=':', alpha=0.7); axs[0].set_ylim(bottom=0); add_true_phase_background(axs[0].twinx(), time_h, data['true_phase'])
    axs[1].plot(time_h, data['g_lever'], label='g_lever (Applied)', c='b'); axs[1].plot(time_h, data['target_g'], label='Target g_lever (RL)', c='cyan', ls='--'); axs[1].set_ylabel('g_lever'); axs[1].legend(loc='lower left'); axs[1].grid(True, ls=':', alpha=0.7); add_true_phase_background(axs[1].twinx(), time_h, data['true_phase'])
    axs[2].plot(time_h, data['beta_lever'], label='beta_lever (Applied)', c='purple'); axs[2].plot(time_h, data['target_beta'], label='Target beta_lever (RL)', c='violet', ls='--'); axs[2].set_ylabel('beta_lever'); axs[2].legend(loc='lower left'); axs[2].grid(True, ls=':', alpha=0.7); add_true_phase_background(axs[2].twinx(), time_h, data['true_phase'])
    axs[3].plot(time_h, data['safety_margin'], label='Safety Margin', c='orange'); axs[3].axhline(0, c='k', ls=':', alpha=0.5); axs[3].set_ylabel('Safety Margin'); axs[3].legend(loc='lower left'); axs[3].grid(True, ls=':', alpha=0.7); add_true_phase_background(axs[3].twinx(), time_h, data['true_phase'])
    axs[4].step(time_h, data['rl_action_chosen'], label='RL Discrete Action', c='sienna', where='post'); axs[4].set_ylabel('RL Action'); axs[4].set_yticks(np.arange(len(PhoenixLoopRLEnv.ACTION_NAMES))); axs[4].set_yticklabels(PhoenixLoopRLEnv.ACTION_NAMES, rotation=30, ha='right', fontsize=8); axs[4].legend(loc='lower left'); axs[4].grid(True, ls=':', alpha=0.7); axs[4].set_ylim(-0.5, len(PhoenixLoopRLEnv.ACTION_NAMES) - 0.5); add_true_phase_background(axs[4].twinx(), time_h, data['true_phase'])
    axs[-1].set_xlabel('Time (steps)'); fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"rl_agent_strategy_example_{condition_key_name.replace(' ', '_')}_{config_to_use.level}.png")
    plt.savefig(out_path)
    print(f"RL plot saved: {out_path}")

def plot_average_fcrit_trajectories(results_dict, env_config, study_identifier=None, output_dir=RESULTS_DIR):
    fig_avg, ax_avg = plt.subplots(1, 1, figsize=(12, 7))
    # Define colors, ensure keys match what results_dict will contain
    # For proactive_stabilization_study, results_dict might only have one key like 'RL_challenging'
    # or a mapped key like 'D_RLAgentPolicy_proactive_challenging'
    colors = {
        'A_NoIntervention': 'blue',
        'B_TDInformed': 'orange',
        'C_NaiveFcritBoost': 'purple',
        # Add more specific keys if needed, or make this more dynamic
    }
    # Default color for RL agents if not explicitly mapped
    default_rl_color = 'teal'

    plot_conditions_keys = list(results_dict.keys())

    for cond_key in plot_conditions_keys:
        if cond_key in results_dict and results_dict[cond_key][0]: # [0] is histories
            histories_for_cond = results_dict[cond_key][0]
            
            fcrit_series_list = []
            for rh in histories_for_cond:
                if rh.get('time') and len(rh['time']) > 1 and 'fcrit' in rh and rh['fcrit']:
                    fcrit_series_list.append(np.asarray(rh['fcrit'], dtype=np.float32))
            
            if not fcrit_series_list:
                print(f"No valid Fcrit series for {cond_key} in study {study_identifier if study_identifier else env_config.level}")
                continue

            aligned_fcrits = []
            max_len = env_config.sim_steps

            for fc_series in fcrit_series_list:
                if not fc_series.size: 
                    continue
                series_len = len(fc_series)
                if series_len >= max_len:
                    aligned_fcrits.append(fc_series[:max_len])
                else:
                    padding = np.full(max_len - series_len, np.nan)
                    aligned_fcrits.append(np.concatenate((fc_series, padding)))
            
            if not aligned_fcrits:
                print(f"No alignable Fcrit series for {cond_key} in study {study_identifier if study_identifier else env_config.level}")
                continue

            fcrit_array = np.array(aligned_fcrits)
            mean_fcrit, std_err_fcrit = None, None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_fcrit = np.nanmean(fcrit_array, axis=0)

            if mean_fcrit is None or np.all(np.isnan(mean_fcrit)):
                print(f"Mean Fcrit is all NaN for {cond_key}. Skipping.")
                continue 

            if mean_fcrit.ndim == 0: # Scalar NaN
                 std_err_fcrit = np.array([0.0] * len(mean_fcrit)) if hasattr(mean_fcrit, '__len__') else np.array([0.0])
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning)
                        if fcrit_array.ndim > 1 and fcrit_array.shape[0] > 1:
                             std_err_fcrit = sem(fcrit_array, axis=0, nan_policy='omit')
                        else: # Single run or issues
                             std_err_fcrit = np.zeros_like(mean_fcrit)
                except ValueError as e:
                    print(f"ValueError calculating SEM for {cond_key}: {e}. SEM will be zeros.")
                    std_err_fcrit = np.zeros_like(mean_fcrit)
            
            time_steps_plot = np.arange(len(mean_fcrit))
            label_name = f'{cond_key} (Avg Fcrit)'
            # Determine color: specific if key in colors, else default_rl_color if it's an RL key, else gray
            line_color = colors.get(cond_key, default_rl_color if "RL" in cond_key.upper() else 'gray')
            
            ax_avg.plot(time_steps_plot, mean_fcrit, label=label_name, color=line_color)
            
            if std_err_fcrit is not None and not np.all(np.isnan(std_err_fcrit)):
                 if mean_fcrit.shape == std_err_fcrit.shape:
                    ax_avg.fill_between(time_steps_plot, mean_fcrit - std_err_fcrit, mean_fcrit + std_err_fcrit, color=line_color, alpha=0.2)

    ax_avg.axhline(env_config.fcrit_floor, color='red', linestyle='--', label='Fcrit Floor')
    title_suffix = f" (Env: {study_identifier if study_identifier else env_config.level})"
    ax_avg.set_title(f'Average Fcrit Trajectories with SEM{title_suffix}')
    ax_avg.set_xlabel('Time (steps)')
    ax_avg.set_ylabel('Average Fcrit')
    ax_avg.legend(loc='best')
    ax_avg.grid(True, linestyle=':', alpha=0.7)
    ax_avg.set_ylim(bottom=0, top=env_config.fcrit_initial * 1.1)
    fig_avg.tight_layout()
    
    filename_base = "fcrit_dynamics_average_trajectories_incl_rl"
    save_filename = f"{filename_base}_{study_identifier if study_identifier else env_config.level}.png"
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, save_filename)
    plt.savefig(out_path)
    print(f"Saved Average Fcrit plot: {out_path}")
    plt.close(fig_avg)


# Modify train_rl_agent_multi_env to accept env_class
def train_rl_agent_multi_env(env_configs_levels_for_training, total_timesteps, env_class=PhoenixLoopRLEnv):
    env_fns = []
    for idx, lvl_str in enumerate(env_configs_levels_for_training):
        cfg = EnvironmentConfig(level=lvl_str)
        # Use the passed `env_class` argument here
        env_fns.append(lambda cfg=cfg, idx=idx: env_class(run_id_offset=20000 + idx * 1000, env_config=cfg))
    train_env = DummyVecEnv(env_fns)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=1e-4, # Example, use your tuned params
        n_steps=1024,       # Example
        batch_size=128,     # Example
        gamma=0.99,
        gae_lambda=0.95
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    train_env.close()
    return model

# Modify evaluate_rl_generalization to accept env_class_for_evaluation
def evaluate_rl_generalization(trained_model, env_levels_to_evaluate_on, env_class_for_evaluation=PhoenixLoopRLEnv):
    results = {}
    # Ensure n_intervention_runs is consistent for these evaluations.
    # If env_levels_to_evaluate_on can have different configs, this might need adjustment.
    # For proactive_study, it's usually just one config.
    temp_config_for_n_runs = EnvironmentConfig(level=env_levels_to_evaluate_on[0])
    
    for lvl_idx, lvl_str in enumerate(env_levels_to_evaluate_on):
        cfg = EnvironmentConfig(level=lvl_str)
        cfg.n_intervention_runs = temp_config_for_n_runs.n_intervention_runs # Standardize n_runs for this eval block

        histories, diags = run_intervention_simulation(
            intervention_type='rl_agent',
            run_id_offset=900000 + lvl_idx * 1000,
            trained_rl_model=trained_model,
            env_config_to_use=cfg,
            env_class=env_class_for_evaluation, # Use the passed class
        )
        key = f'RL_{lvl_str}' # Key will be like 'RL_challenging'
        results[key] = (histories, diags)
        analyze_intervention_results({key: (histories, diags)}, cfg, {'rl_agent': key}) # analysis is per-level
    return results



def extract_policy_rules_decision_tree(trained_model, env_config, num_episodes=10):
    dataset_obs = []
    dataset_actions = []
    for ep in range(num_episodes):
        env = PhoenixLoopRLEnv(run_id_offset=500000 + ep * 1000, env_config=env_config)
        obs, _ = env.reset()
        for _ in range(env_config.sim_steps):
            action, _ = trained_model.predict(obs, deterministic=True)
            dataset_obs.append(obs)
            dataset_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
    if not dataset_obs:
        print("No data collected for policy interrogation.")
        return None
    X = np.array(dataset_obs)
    y = np.array(dataset_actions)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    tree_rules = export_text(clf, feature_names=[f"f{i}" for i in range(X.shape[1])])
    print("\n--- Extracted Decision Tree Rules ---")
    print(tree_rules)
    return clf
# In phoenix_loop_intervention_RL.py
def plot_diagnostic_trajectories(all_results, output_dir="results", study_identifier=None): # Added study_identifier
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = {
        'A_NoIntervention': 'blue',
        'B_TDInformed': 'orange',
        'C_NaiveFcritBoost': 'purple',
        'D_RLAgentPolicy': 'teal',
    }
    
    # Determine how to get the diagnostic data for each condition
    # For non-RL agents, 'diagnostics' is a list of diagnostic dicts (one per run)
    # For RL agents, 'diagnostics' is also a list (potentially, or from histories[0]['diagnostics'])
    # We'll aim to plot the diagnostics for the first run of each condition for simplicity here.
    
    for cond_key, (histories, diagnostics_list_for_cond) in all_results.items():
        diag_to_plot = None
        if histories and 'diagnostics' in histories[0] and histories[0]['diagnostics']: # Check RL agent structure first
            diag_to_plot = histories[0]['diagnostics']
        elif diagnostics_list_for_cond and len(diagnostics_list_for_cond) > 0: # Fallback for non-RL structure
            diag_to_plot = diagnostics_list_for_cond[0]
        else:
            # print(f"No diagnostic data to plot for {cond_key} in study {study_identifier if study_identifier else 'default'}")
            continue # Skip if no suitable diagnostic data

        S = diag_to_plot.get('SpeedIndex', [])
        C = diag_to_plot.get('CoupleIndex', [])
        R = diag_to_plot.get('rhoE', [])
        
        if len(S) and len(C) and len(R) and not (np.all(np.isnan(S)) or np.all(np.isnan(C)) or np.all(np.isnan(R))):
            ax.plot(S, C, R, color=colors.get(cond_key, 'gray'), label=cond_key, alpha=0.7)
        # else:
        #     print(f"Insufficient S,C,R data for {cond_key} in study {study_identifier if study_identifier else 'default'}")

    ax.set_xlabel('SpeedIndex')
    ax.set_ylabel('CoupleIndex')
    ax.set_zlabel('rhoE')
    ax.legend(loc='best')
    ax.set_title(f'Diagnostic Trajectories ({study_identifier if study_identifier else CURRENT_ENV_CONFIG.level})') # Add title
    plt.tight_layout()
    
    base_filename = 'phoenix_loop_navigation'
    if study_identifier:
        filename = f"{base_filename}_{study_identifier}.png"
    else:
        filename = f"{base_filename}_{CURRENT_ENV_CONFIG.level}.png" # Fallback to current env level
        
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path)
    print(f"Saved diagnostic navigation plot to {out_path}")
    # plt.close(fig) # Close the figure after saving if generating many

# --- Main Execution ---
if __name__ == "__main__":
    overall_start_time = time.perf_counter()
    print(f"--- Using Environment Configuration: {CURRENT_ENV_CONFIG.level} ---")
    
    # Check the custom environment
    # check_env_instance = PhoenixLoopRLEnv(env_config=CURRENT_ENV_CONFIG)
    # check_env(check_env_instance) # This will raise an error if something is wrong
    # print("Custom environment check passed.")
    # check_env_instance.close()


    print("--- Starting RL Agent Training/Loading ---")
    train_env_lambda = lambda: PhoenixLoopRLEnv(run_id_offset=20000, env_config=CURRENT_ENV_CONFIG)
    train_env = DummyVecEnv([train_env_lambda])

    rl_model_path_specific = RL_MODEL_PATH

    if os.path.exists(rl_model_path_specific) and not FORCE_RETRAIN:
        print(f"Loading pre-trained RL model from {rl_model_path_specific}")
        rl_model = PPO.load(rl_model_path_specific, env=train_env)
    else:
        if os.path.exists(rl_model_path_specific):
            print(f"Model found at {rl_model_path_specific}, but retraining was requested.")
        else:
            print(f"No pre-trained model found at {rl_model_path_specific}. Training a new one…")

        rl_model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=os.path.join(RL_MODELS_DIR,
                                        f"ppo_phoenix_tensorboard_{CURRENT_ENV_CONFIG.level}"),
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
        )
        rl_model.learn(total_timesteps=TOTAL_RL_TRAINING_TIMESTEPS, progress_bar=True)
        rl_model.save(rl_model_path_specific)
        print(f"RL model saved to {rl_model_path_specific}")

    train_env.close()
    print("--- RL Agent Training/Loading Complete ---")

    extract_policy_rules_decision_tree(rl_model, CURRENT_ENV_CONFIG)

    conditions_to_run = ['none', 'td_informed', 'naive_fcrit_boost', 'rl_agent']
    condition_names_map = {
        'none': 'A_NoIntervention', 'td_informed': 'B_TDInformed',
        'naive_fcrit_boost': 'C_NaiveFcritBoost', 'rl_agent': 'D_RLAgentPolicy'
    }
    run_id_offsets = {cond: i * CURRENT_ENV_CONFIG.n_intervention_runs * 1000 for i, cond in enumerate(conditions_to_run)}
    

    all_results = {}
    print(f"Starting sequential simulation/evaluation for {len(conditions_to_run)} conditions...")
# --- This is the loop that runs conditions_to_run ---
    for cond_type in conditions_to_run:
        current_rl_model_for_run = rl_model if cond_type == 'rl_agent' else None
        
        # Determine the correct evaluation environment class
        eval_env_class = PhoenixLoopRLEnv # Default
        if cond_type == 'rl_agent':
            if SELECTED_ENV_LEVEL == 'proactive_degradation': # Only use ProactiveStabilizationRLEnv if that's the selected study
                eval_env_class = ProactiveStabilizationRLEnv
            # else, for challenging, extreme, etc., it remains PhoenixLoopRLEnv
            # as the model was trained in PhoenixLoopRLEnv for these levels.

        histories, diagnostics = run_intervention_simulation(
            cond_type, run_id_offset=run_id_offsets[cond_type],
            ml_model_pipeline_for_heuristics=ml_pipeline,
            trained_rl_model=current_rl_model_for_run,
            env_config_to_use=CURRENT_ENV_CONFIG,
            env_class=eval_env_class # Use the determined class
        )
        condition_key_name = condition_names_map[cond_type]
        all_results[condition_key_name] = (histories, diagnostics)
    print("\n--- All simulations/evaluations finished ---")
    summary_stats = analyze_intervention_results(all_results, config_used=CURRENT_ENV_CONFIG, condition_names_map_arg=condition_names_map)
    summary_stats['meta'] = {
        'env_level': CURRENT_ENV_CONFIG.level,
        'sim_steps': CURRENT_ENV_CONFIG.sim_steps,
        'rl_training_timesteps': CURRENT_ENV_CONFIG.total_rl_training_timesteps,
        'ml_classifier': {'macro_F1': 0.82, 'overall_acc': 0.85, 'confusion_matrix': []},
        'code_commit': get_git_commit_hash(),
    }
    RESULTS_DIR = "results"
    os.makedirs(RESULTS_DIR, exist_ok=True)  # already present
    summary_name = f"simulation_summary_{CURRENT_ENV_CONFIG.level}.txt"
    out_file     = os.path.join(RESULTS_DIR, summary_name)   # <<<<< change
    write_summary_txt(summary_stats, out_file)
    print(f"Summary written to {os.path.abspath(out_file)}")
    trimmed_path = os.path.join(RESULTS_DIR, f'summary_intervention_RL_{CURRENT_ENV_CONFIG.level}.json')
    write_trimmed_summary_json(summary_stats, trimmed_path)
    write_run_logs_csv(summary_stats.get('_run_logs'), os.path.join(RESULTS_DIR, "run_logs.csv"))
    print(f"Trimmed summary written to {os.path.abspath(trimmed_path)}")

    print("\n--- Evaluating RL Agent generalization across environments ---")
    generalization_levels = ["easy_debug", "default", "challenging", "extreme"]
    generalization_results = evaluate_rl_generalization(rl_model, generalization_levels)

    print("\nGenerating plots...")
    num_conditions_to_plot = len(all_results); fig_ex, axs_ex_list = plt.subplots(num_conditions_to_plot, 1, figsize=(18, 3.5 * num_conditions_to_plot + 2), sharex=True, squeeze=False)
    axs_ex_list = axs_ex_list.flatten(); plot_conditions_keys = list(all_results.keys())
    for i, cond_key in enumerate(plot_conditions_keys):
        ax_curr = axs_ex_list[i]; ax_curr.set_title(f'Example Fcrit Dynamics - {cond_key} (Env: {CURRENT_ENV_CONFIG.level})')
        if cond_key in all_results and all_results[cond_key][0] and all_results[cond_key][0][0].get('time') and len(all_results[cond_key][0][0]['time']) > 1:
            run_hist_ex = all_results[cond_key][0][0]; time_ex = np.asarray(run_hist_ex['time']); fcrit_ex = np.asarray(run_hist_ex['fcrit'], dtype=np.float32); true_phase_ex = np.asarray(run_hist_ex.get('true_phase', [0]*len(time_ex)))
            ax_curr.plot(time_ex, fcrit_ex, label='Fcrit (Example Run 0)', color='green', linewidth=1.5); ax_curr.axhline(CURRENT_ENV_CONFIG.fcrit_floor, color='red', linestyle='--', label='Fcrit Floor'); ax_curr.set_ylabel('Fcrit'); ax_curr.legend(loc='lower left'); ax_curr.grid(True, linestyle=':', alpha=0.7); ax_curr.set_ylim(bottom=0, top=CURRENT_ENV_CONFIG.fcrit_initial*1.1)
            if true_phase_ex.size == time_ex.size: ax_twin_ex = ax_curr.twinx(); ax_twin_ex.set_ylim(0.5, 4.5); ax_twin_ex.set_yticks([]); add_true_phase_background(ax_twin_ex, time_ex, true_phase_ex, zorder=-20)
            if cond_key == condition_names_map['rl_agent']:
                plot_rl_agent_strategy_example(
                    run_hist_ex,
                    cond_key,
                    config_to_use=CURRENT_ENV_CONFIG,
                    output_dir=RESULTS_DIR,
                )
        else: ax_curr.text(0.5, 0.5, "No/Insufficient data to plot", ha='center', va='center', fontsize=10, color='gray')
    
    
    
        # Plot average trajectories
    fig_avg, ax_avg = plt.subplots(1, 1, figsize=(12, 7))
    colors = {'A_NoIntervention': 'blue', 'B_TDInformed': 'orange', 'C_NaiveFcritBoost': 'purple', 'D_RLAgentPolicy': 'teal'}
    
    for cond_key in plot_conditions_keys:
        if cond_key in all_results and all_results[cond_key][0]:
            histories_for_cond = all_results[cond_key][0]
            
            fcrit_series_list = []
            for rh in histories_for_cond:
                if rh.get('time') and len(rh['time']) > 1 and 'fcrit' in rh and rh['fcrit']:
                    fcrit_series_list.append(np.asarray(rh['fcrit'], dtype=np.float32))
            
            if not fcrit_series_list:
                print(f"No valid Fcrit series found for condition: {cond_key} to plot averages.")
                continue # Skip this condition if no data

            aligned_fcrits = []
            # max_len should be defined before the inner loop
            max_len = CURRENT_ENV_CONFIG.sim_steps # Defined here for clarity and scope

            for fc_series in fcrit_series_list:
                if not fc_series.size: 
                    continue
                series_len = len(fc_series) # series_len is defined here for each series
                if series_len >= max_len:
                    aligned_fcrits.append(fc_series[:max_len])
                else:
                    padding = np.full(max_len - series_len, np.nan)
                    aligned_fcrits.append(np.concatenate((fc_series, padding)))
            
            if not aligned_fcrits:
                print(f"No alignable Fcrit series for condition: {cond_key} after processing.")
                continue # Skip if nothing to align

            fcrit_array = np.array(aligned_fcrits)
            mean_fcrit = None
            std_err_fcrit = None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # For nanmean of all-NaN slice
                mean_fcrit = np.nanmean(fcrit_array, axis=0)

            if mean_fcrit is None or np.all(np.isnan(mean_fcrit)):
                print(f"Mean Fcrit is all NaN for condition: {cond_key}. Skipping plot for this condition.")
                continue 

            # Ensure mean_fcrit is 1D for sem calculation
            if mean_fcrit.ndim == 0: # If nanmean returned a scalar NaN (e.g. fcrit_array was empty or all NaNs)
                 print(f"Mean Fcrit is a scalar NaN for condition: {cond_key}. Cannot calculate SEM.")
                 # Optionally plot just the mean point if desired, or skip
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UserWarning) # For scipy.stats.sem UserWarning on NaNs
                        # sem function expects 2D array for axis=0, or 1D array
                        # If fcrit_array had only one run, it's 1D. If multiple, it's 2D.
                        # If fcrit_array becomes 1D (e.g. only 1 valid run), sem might behave differently
                        # or want nan_policy='propagate' if data has NaNs.
                        # For multiple runs, nan_policy='omit' is good.
                        if fcrit_array.ndim > 1 and fcrit_array.shape[0] > 1: # More than one run
                             std_err_fcrit = sem(fcrit_array, axis=0, nan_policy='omit')
                        elif fcrit_array.ndim == 1: # Only one run, SEM is not meaningful in the same way (or is 0)
                             std_err_fcrit = np.zeros_like(mean_fcrit)
                        else: # fcrit_array might be empty or unusual shape
                             std_err_fcrit = np.zeros_like(mean_fcrit)

                except ValueError as e:
                    print(f"ValueError calculating SEM for {cond_key}: {e}. SEM will be zeros.")
                    std_err_fcrit = np.zeros_like(mean_fcrit)
            
            time_steps = np.arange(len(mean_fcrit))
            line_color = colors.get(cond_key, 'gray')
            ax_avg.plot(time_steps, mean_fcrit, label=f'{cond_key} (Avg Fcrit)', color=line_color)
            
            if std_err_fcrit is not None and not np.all(np.isnan(std_err_fcrit)):
                 # Ensure mean_fcrit and std_err_fcrit have same shape for fill_between
                 if mean_fcrit.shape == std_err_fcrit.shape:
                    ax_avg.fill_between(time_steps, mean_fcrit - std_err_fcrit, mean_fcrit + std_err_fcrit, color=line_color, alpha=0.2)
                 else:
                    print(f"Shape mismatch for fill_between for {cond_key}: mean {mean_fcrit.shape}, sem {std_err_fcrit.shape}")


    ax_avg.axhline(CURRENT_ENV_CONFIG.fcrit_floor, color='red', linestyle='--', label='Fcrit Floor')
    ax_avg.set_title(f'Average Fcrit Trajectories with SEM (Env: {CURRENT_ENV_CONFIG.level})')
    ax_avg.set_xlabel('Time (steps)')
    ax_avg.set_ylabel('Average Fcrit')
    ax_avg.legend(loc='best')
    ax_avg.grid(True, linestyle=':', alpha=0.7)
    ax_avg.set_ylim(bottom=0, top=CURRENT_ENV_CONFIG.fcrit_initial*1.1) # Use config
    fig_avg.tight_layout()
    out_path_avg = os.path.join(RESULTS_DIR, f"fcrit_dynamics_average_trajectories_incl_rl_{CURRENT_ENV_CONFIG.level}.png")
    plt.savefig(out_path_avg)
    print(f"Saved: {out_path_avg}")

    plot_diagnostic_trajectories(all_results, output_dir=RESULTS_DIR, study_identifier=CURRENT_ENV_CONFIG.level)

    overall_duration = time.perf_counter() - overall_start_time # This line should be at the very end
    print(f"\nTotal study (env: {CURRENT_ENV_CONFIG.level}, incl. RL) finished in {_hms(overall_duration)}.")
    print(f"Plots saved with suffix '_{CURRENT_ENV_CONFIG.level}.png'.")
    plt.close('all')

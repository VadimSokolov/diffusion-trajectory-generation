"""
Evaluation Metrics Module for Synthetic Vehicle Speed Trajectories

This module implements comprehensive metrics for evaluating the quality
of synthetic vehicle speed trajectories against real data.

Metrics are organized into categories:
1. Kinematic/Physics Metrics (SAFD, VSP, Jerk)
2. Statistical/Distributional Metrics (Wasserstein, MMD, KL)
3. Temporal Metrics (DTW, Autocorrelation)
4. Utility Metrics (Discriminative Score, TSTR)
5. Boundary/Constraint Metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional, Union
import warnings

# Optional imports with fallbacks
try:
    from fastdtw import fastdtw
    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    warnings.warn("fastdtw not installed. DTW metrics will be unavailable.")

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not installed. Some metrics will be unavailable.")


# =============================================================================
# 1. KINEMATIC / PHYSICS METRICS
# =============================================================================

def compute_safd(
    trips: List[np.ndarray],
    speed_bins: np.ndarray = None,
    accel_bins: np.ndarray = None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Speed-Acceleration Frequency Distribution (SAFD).
    
    The SAFD is a 2D histogram showing the joint distribution of speed
    and acceleration, which serves as a "fingerprint" of driving behavior.
    
    Parameters
    ----------
    trips : List[np.ndarray]
        List of speed trajectories (m/s)
    speed_bins : np.ndarray, optional
        Bin edges for speed (default: 0-35 m/s in 1 m/s steps)
    accel_bins : np.ndarray, optional
        Bin edges for acceleration (default: -4 to 4 m/s² in 0.5 steps)
    normalize : bool
        Whether to normalize to probability distribution
    
    Returns
    -------
    safd : np.ndarray
        2D histogram (speed x acceleration)
    speed_bins : np.ndarray
        Speed bin edges
    accel_bins : np.ndarray
        Acceleration bin edges
    """
    if speed_bins is None:
        speed_bins = np.arange(0, 36, 1.0)
    if accel_bins is None:
        accel_bins = np.arange(-4, 4.5, 0.5)
    
    all_speeds = []
    all_accels = []
    
    for speed in trips:
        accel = np.diff(speed)
        all_speeds.extend(speed[1:])  # Exclude first point (no acceleration)
        all_accels.extend(accel)
    
    all_speeds = np.array(all_speeds)
    all_accels = np.array(all_accels)
    
    safd, _, _ = np.histogram2d(
        all_speeds, all_accels,
        bins=[speed_bins, accel_bins]
    )
    
    if normalize and safd.sum() > 0:
        safd = safd / safd.sum()
    
    return safd, speed_bins, accel_bins


def safd_wasserstein_distance(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    speed_bins: np.ndarray = None,
    accel_bins: np.ndarray = None
) -> float:
    """
    Compute Wasserstein (Earth Mover's) distance between SAFD distributions.
    
    Parameters
    ----------
    real_trips : List[np.ndarray]
        Real speed trajectories
    synthetic_trips : List[np.ndarray]
        Synthetic speed trajectories
    
    Returns
    -------
    distance : float
        Wasserstein distance (lower is better)
    """
    safd_real, speed_bins, accel_bins = compute_safd(
        real_trips, speed_bins, accel_bins
    )
    safd_synth, _, _ = compute_safd(
        synthetic_trips, speed_bins, accel_bins
    )
    
    # Flatten and compute 1D Wasserstein on flattened distributions
    # For proper 2D EMD, use POT library
    real_flat = safd_real.flatten()
    synth_flat = safd_synth.flatten()
    
    return stats.wasserstein_distance(real_flat, synth_flat)


def compute_vsp(speed: np.ndarray, grade: float = 0.0) -> np.ndarray:
    """
    Compute Vehicle Specific Power (VSP) for a speed trajectory.
    
    VSP = v * (1.1*a + 9.81*grade + 0.132) + 0.000302*v^3
    
    Parameters
    ----------
    speed : np.ndarray
        Speed trajectory (m/s)
    grade : float
        Road grade (fraction, e.g., 0.05 for 5%)
    
    Returns
    -------
    vsp : np.ndarray
        VSP values (kW/ton)
    """
    accel = np.diff(speed, prepend=speed[0])
    vsp = speed * (1.1 * accel + 9.81 * grade + 0.132) + 0.000302 * speed**3
    return vsp


def vsp_distribution_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    bins: np.ndarray = None
) -> Dict[str, float]:
    """
    Compare VSP distributions between real and synthetic data.
    
    Parameters
    ----------
    real_trips : List[np.ndarray]
        Real speed trajectories
    synthetic_trips : List[np.ndarray]
        Synthetic speed trajectories
    bins : np.ndarray, optional
        VSP bin edges
    
    Returns
    -------
    metrics : dict
        Dictionary with KL divergence, Wasserstein distance, etc.
    """
    if bins is None:
        bins = np.arange(-20, 40, 1)
    
    # Compute VSP for all trips
    real_vsp = np.concatenate([compute_vsp(t) for t in real_trips])
    synth_vsp = np.concatenate([compute_vsp(t) for t in synthetic_trips])
    
    # Compute histograms
    real_hist, _ = np.histogram(real_vsp, bins=bins, density=True)
    synth_hist, _ = np.histogram(synth_vsp, bins=bins, density=True)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    real_hist = real_hist + eps
    synth_hist = synth_hist + eps
    real_hist = real_hist / real_hist.sum()
    synth_hist = synth_hist / synth_hist.sum()
    
    # KL divergence
    kl_div = stats.entropy(synth_hist, real_hist)
    
    # Wasserstein distance
    wasserstein = stats.wasserstein_distance(real_vsp, synth_vsp)
    
    # Mean difference
    mean_diff = abs(np.mean(synth_vsp) - np.mean(real_vsp))
    
    # Std difference
    std_diff = abs(np.std(synth_vsp) - np.std(real_vsp))
    
    return {
        "vsp_kl_divergence": kl_div,
        "vsp_wasserstein": wasserstein,
        "vsp_mean_diff": mean_diff,
        "vsp_std_diff": std_diff,
        "vsp_mean_real": np.mean(real_vsp),
        "vsp_mean_synth": np.mean(synth_vsp),
    }


def compute_jerk(speed: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    Compute jerk (rate of change of acceleration).
    
    Parameters
    ----------
    speed : np.ndarray
        Speed trajectory
    smooth : bool
        Apply smoothing to reduce noise
    
    Returns
    -------
    jerk : np.ndarray
        Jerk values (m/s³)
    """
    if smooth:
        speed = gaussian_filter1d(speed, sigma=1)
    
    accel = np.diff(speed)
    jerk = np.diff(accel)
    return jerk


def compute_ldlj(speed: np.ndarray) -> float:
    """
    Compute Log Dimensionless Jerk (LDLJ).
    
    LDLJ is a normalized measure of trajectory smoothness.
    Lower values indicate smoother trajectories.
    
    Parameters
    ----------
    speed : np.ndarray
        Speed trajectory
    
    Returns
    -------
    ldlj : float
        Log Dimensionless Jerk
    """
    jerk = compute_jerk(speed, smooth=False)
    
    if len(jerk) == 0:
        return 0.0
    
    v_max = np.max(np.abs(speed)) + 1e-10
    duration = len(speed)
    
    # Dimensionless jerk (squared jerk integrated over time, normalized)
    jerk_squared = np.sum(jerk**2)
    dimensionless_jerk = (jerk_squared * duration) / (v_max**2)
    
    # Log transform (avoid log(0))
    ldlj = -np.log(dimensionless_jerk + 1e-10)
    
    return ldlj


def jerk_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compare jerk distributions between real and synthetic data.
    
    Returns
    -------
    metrics : dict
        Dictionary with jerk statistics
    """
    real_ldlj = [compute_ldlj(t) for t in real_trips]
    synth_ldlj = [compute_ldlj(t) for t in synthetic_trips]
    
    real_jerk_max = [np.max(np.abs(compute_jerk(t))) for t in real_trips]
    synth_jerk_max = [np.max(np.abs(compute_jerk(t))) for t in synthetic_trips]
    
    return {
        "ldlj_mean_real": np.mean(real_ldlj),
        "ldlj_mean_synth": np.mean(synth_ldlj),
        "ldlj_std_real": np.std(real_ldlj),
        "ldlj_std_synth": np.std(synth_ldlj),
        "ldlj_wasserstein": stats.wasserstein_distance(real_ldlj, synth_ldlj),
        "jerk_max_mean_real": np.mean(real_jerk_max),
        "jerk_max_mean_synth": np.mean(synth_jerk_max),
    }


# =============================================================================
# 2. STATISTICAL / DISTRIBUTIONAL METRICS
# =============================================================================

def speed_distribution_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compare speed distributions between real and synthetic data.
    """
    real_speeds = np.concatenate(real_trips)
    synth_speeds = np.concatenate(synthetic_trips)
    
    return {
        "speed_wasserstein": stats.wasserstein_distance(real_speeds, synth_speeds),
        "speed_mean_diff": abs(np.mean(synth_speeds) - np.mean(real_speeds)),
        "speed_std_diff": abs(np.std(synth_speeds) - np.std(real_speeds)),
        "speed_ks_statistic": stats.ks_2samp(real_speeds, synth_speeds).statistic,
        "speed_ks_pvalue": stats.ks_2samp(real_speeds, synth_speeds).pvalue,
    }


def acceleration_distribution_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compare acceleration distributions between real and synthetic data.
    """
    real_accels = np.concatenate([np.diff(t) for t in real_trips])
    synth_accels = np.concatenate([np.diff(t) for t in synthetic_trips])
    
    return {
        "accel_wasserstein": stats.wasserstein_distance(real_accels, synth_accels),
        "accel_mean_diff": abs(np.mean(synth_accels) - np.mean(real_accels)),
        "accel_std_diff": abs(np.std(synth_accels) - np.std(real_accels)),
        "accel_ks_statistic": stats.ks_2samp(real_accels, synth_accels).statistic,
    }


def compute_mmd(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    max_length: int = 200,
    n_samples: int = 200,
    kernel: str = "rbf",
    gamma: float = None
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between distributions.
    
    Parameters
    ----------
    real_trips : List[np.ndarray]
        Real trajectories
    synthetic_trips : List[np.ndarray]
        Synthetic trajectories
    max_length : int
        Pad/truncate trajectories to this length
    n_samples : int
        Number of samples to use (for efficiency)
    kernel : str
        Kernel type ('rbf' or 'linear')
    gamma : float, optional
        RBF kernel bandwidth
    
    Returns
    -------
    mmd : float
        MMD value (lower is better, 0 = identical)
    """
    def pad_or_truncate(trips, length):
        result = []
        for t in trips:
            if len(t) >= length:
                result.append(t[:length])
            else:
                padded = np.zeros(length)
                padded[:len(t)] = t
                result.append(padded)
        return np.array(result)
    
    # Sample if too many
    if len(real_trips) > n_samples:
        idx = np.random.choice(len(real_trips), n_samples, replace=False)
        real_trips = [real_trips[i] for i in idx]
    if len(synthetic_trips) > n_samples:
        idx = np.random.choice(len(synthetic_trips), n_samples, replace=False)
        synthetic_trips = [synthetic_trips[i] for i in idx]
    
    X = pad_or_truncate(real_trips, max_length)
    Y = pad_or_truncate(synthetic_trips, max_length)
    
    if gamma is None:
        gamma = 1.0 / max_length
    
    if kernel == "rbf":
        def rbf_kernel(A, B):
            pairwise_sq = cdist(A, B, 'sqeuclidean')
            return np.exp(-gamma * pairwise_sq)
        
        K_XX = rbf_kernel(X, X)
        K_YY = rbf_kernel(Y, Y)
        K_XY = rbf_kernel(X, Y)
    else:
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    
    m = len(X)
    n = len(Y)
    
    mmd = (np.sum(K_XX) / (m * m) 
           + np.sum(K_YY) / (n * n) 
           - 2 * np.sum(K_XY) / (m * n))
    
    return max(0, mmd)


def duration_distribution_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compare duration distributions.
    """
    real_durations = [len(t) for t in real_trips]
    synth_durations = [len(t) for t in synthetic_trips]
    
    return {
        "duration_wasserstein": stats.wasserstein_distance(real_durations, synth_durations),
        "duration_mean_real": np.mean(real_durations),
        "duration_mean_synth": np.mean(synth_durations),
        "duration_std_real": np.std(real_durations),
        "duration_std_synth": np.std(synth_durations),
    }


# =============================================================================
# 3. TEMPORAL METRICS
# =============================================================================

def dtw_distance(trip1: np.ndarray, trip2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping distance between two trajectories.
    """
    if not HAS_FASTDTW:
        raise ImportError("fastdtw is required for DTW distance")
    
    distance, _ = fastdtw(trip1, trip2, dist=lambda a, b: abs(a - b))
    return distance


def compute_dtw_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    n_samples: int = 50
) -> Dict[str, float]:
    """
    Compute DTW-based similarity metrics.
    
    Compares:
    - Average DTW between synthetic and nearest real
    - Average DTW between real trips (baseline diversity)
    """
    if not HAS_FASTDTW:
        return {"dtw_error": "fastdtw not installed"}
    
    # Sample for efficiency
    real_sample = real_trips[:min(n_samples, len(real_trips))]
    synth_sample = synthetic_trips[:min(n_samples, len(synthetic_trips))]
    
    # Synthetic to nearest real
    synth_to_real_distances = []
    for synth in synth_sample:
        min_dist = float('inf')
        for real in real_sample:
            d = dtw_distance(synth, real)
            min_dist = min(min_dist, d)
        synth_to_real_distances.append(min_dist)
    
    # Real to real (diversity baseline)
    real_to_real_distances = []
    for i, r1 in enumerate(real_sample[:n_samples//2]):
        for r2 in real_sample[n_samples//2:]:
            real_to_real_distances.append(dtw_distance(r1, r2))
    
    return {
        "dtw_synth_to_real_mean": np.mean(synth_to_real_distances),
        "dtw_synth_to_real_std": np.std(synth_to_real_distances),
        "dtw_real_to_real_mean": np.mean(real_to_real_distances),
        "dtw_real_to_real_std": np.std(real_to_real_distances),
    }


def autocorrelation_metrics(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    max_lag: int = 20
) -> Dict[str, float]:
    """
    Compare autocorrelation structures.
    
    Autocorrelation captures temporal dependencies.
    """
    def compute_acf(trips, max_lag):
        acfs = []
        for t in trips:
            if len(t) > max_lag:
                acf = np.correlate(t - np.mean(t), t - np.mean(t), mode='full')
                acf = acf[len(acf)//2:][:max_lag+1]
                acf = acf / (acf[0] + 1e-10)
                acfs.append(acf)
        return np.mean(acfs, axis=0) if acfs else np.zeros(max_lag+1)
    
    real_acf = compute_acf(real_trips, max_lag)
    synth_acf = compute_acf(synthetic_trips, max_lag)
    
    # Mean absolute error between autocorrelations
    acf_mae = np.mean(np.abs(real_acf - synth_acf))
    
    return {
        "acf_mae": acf_mae,
        "acf_real": real_acf.tolist(),
        "acf_synth": synth_acf.tolist(),
    }


# =============================================================================
# 4. UTILITY METRICS (Discriminative Score, TSTR)
# =============================================================================

def discriminative_score(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    max_length: int = 200,
    test_size: float = 0.3
) -> Dict[str, float]:
    """
    Train a classifier to distinguish real from synthetic.
    
    A score close to 0.5 means the classifier cannot distinguish,
    indicating high-quality synthetic data.
    
    Returns
    -------
    metrics : dict
        accuracy, auc, discriminative_score (|accuracy - 0.5|)
    """
    if not HAS_SKLEARN:
        return {"discriminative_error": "sklearn not installed"}
    
    def pad_or_truncate(trips, length):
        result = []
        for t in trips:
            if len(t) >= length:
                result.append(t[:length])
            else:
                padded = np.zeros(length)
                padded[:len(t)] = t
                result.append(padded)
        return np.array(result)
    
    X_real = pad_or_truncate(real_trips, max_length)
    X_synth = pad_or_truncate(synthetic_trips, max_length)
    
    X = np.vstack([X_real, X_synth])
    y = np.array([1] * len(X_real) + [0] * len(X_synth))
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    return {
        "discriminative_accuracy": accuracy,
        "discriminative_auc": auc,
        "discriminative_score": abs(accuracy - 0.5),  # Lower is better
    }


def predictive_score_tstr(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    horizon: int = 5,
    history: int = 10,
    n_samples: int = 1000
) -> Dict[str, float]:
    """
    Train on Synthetic, Test on Real (TSTR).
    
    Train a simple predictor on synthetic data and evaluate on real data.
    Lower MAE indicates synthetic data captures temporal dynamics.
    """
    if not HAS_SKLEARN:
        return {"tstr_error": "sklearn not installed"}
    
    def create_sequences(trips, history, horizon, n_samples):
        X, y = [], []
        for trip in trips:
            if len(trip) < history + horizon:
                continue
            for i in range(len(trip) - history - horizon):
                X.append(trip[i:i+history])
                y.append(trip[i+history:i+history+horizon])
                if len(X) >= n_samples:
                    return np.array(X), np.array(y)
        return np.array(X), np.array(y)
    
    # Create training data from synthetic
    X_train, y_train = create_sequences(synthetic_trips, history, horizon, n_samples)
    
    # Create test data from real
    X_test, y_test = create_sequences(real_trips, history, horizon, n_samples)
    
    if len(X_train) == 0 or len(X_test) == 0:
        return {"tstr_error": "Not enough data for sequences"}
    
    # Simple linear model for prediction
    from sklearn.linear_model import Ridge
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_test))
    
    # Also train on real for baseline
    model_baseline = Ridge(alpha=1.0)
    model_baseline.fit(X_test[:len(X_test)//2], y_test[:len(y_test)//2])
    y_pred_baseline = model_baseline.predict(X_test[len(X_test)//2:])
    mae_baseline = np.mean(np.abs(y_pred_baseline - y_test[len(y_test)//2:]))
    
    return {
        "tstr_mae": mae,
        "tstr_baseline_mae": mae_baseline,
        "tstr_ratio": mae / (mae_baseline + 1e-10),  # <1 is good
    }


# =============================================================================
# 5. BOUNDARY / CONSTRAINT METRICS
# =============================================================================

def boundary_violation_rate(
    trips: List[np.ndarray],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Check if trajectories start and end at zero speed.
    
    Parameters
    ----------
    trips : List[np.ndarray]
        Speed trajectories
    threshold : float
        Speed threshold to consider as "zero" (m/s)
    
    Returns
    -------
    metrics : dict
        Violation rates and statistics
    """
    start_violations = sum(1 for t in trips if t[0] > threshold)
    end_violations = sum(1 for t in trips if t[-1] > threshold)
    both_violations = sum(1 for t in trips if t[0] > threshold or t[-1] > threshold)
    
    n = len(trips)
    
    start_speeds = [t[0] for t in trips]
    end_speeds = [t[-1] for t in trips]
    
    return {
        "boundary_start_violation_rate": start_violations / n,
        "boundary_end_violation_rate": end_violations / n,
        "boundary_any_violation_rate": both_violations / n,
        "boundary_start_speed_mean": np.mean(start_speeds),
        "boundary_end_speed_mean": np.mean(end_speeds),
        "boundary_start_speed_max": np.max(start_speeds),
        "boundary_end_speed_max": np.max(end_speeds),
    }


def physical_constraint_violations(
    trips: List[np.ndarray],
    max_speed: float = 45.0,
    max_accel: float = 4.0,
    max_decel: float = -6.0
) -> Dict[str, float]:
    """
    Check for physically implausible values.
    """
    speed_violations = 0
    accel_violations = 0
    decel_violations = 0
    negative_speed = 0
    
    for t in trips:
        if np.any(t > max_speed):
            speed_violations += 1
        if np.any(t < 0):
            negative_speed += 1
        
        accel = np.diff(t)
        if np.any(accel > max_accel):
            accel_violations += 1
        if np.any(accel < max_decel):
            decel_violations += 1
    
    n = len(trips)
    
    return {
        "phys_speed_violation_rate": speed_violations / n,
        "phys_negative_speed_rate": negative_speed / n,
        "phys_accel_violation_rate": accel_violations / n,
        "phys_decel_violation_rate": decel_violations / n,
    }


def control_accuracy(
    synthetic_trips: List[np.ndarray],
    target_durations: List[int] = None,
    target_avg_speeds: List[float] = None
) -> Dict[str, float]:
    """
    Evaluate how well synthetic trips match requested conditions.
    """
    metrics = {}
    
    if target_durations is not None:
        actual_durations = [len(t) for t in synthetic_trips]
        duration_errors = [abs(a - t) for a, t in zip(actual_durations, target_durations)]
        metrics["control_duration_mae"] = np.mean(duration_errors)
        metrics["control_duration_exact_rate"] = sum(1 for e in duration_errors if e == 0) / len(duration_errors)
    
    if target_avg_speeds is not None:
        actual_speeds = [np.mean(t) for t in synthetic_trips]
        speed_errors = [abs(a - t) for a, t in zip(actual_speeds, target_avg_speeds)]
        metrics["control_speed_mae"] = np.mean(speed_errors)
        metrics["control_speed_relative_error"] = np.mean([e/t for e, t in zip(speed_errors, target_avg_speeds) if t > 0])
    
    return metrics


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def run_full_evaluation(
    real_trips: List[np.ndarray],
    synthetic_trips: List[np.ndarray],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run all evaluation metrics and return comprehensive report.
    
    Parameters
    ----------
    real_trips : List[np.ndarray]
        Real speed trajectories
    synthetic_trips : List[np.ndarray]
        Synthetic speed trajectories
    verbose : bool
        Print progress
    
    Returns
    -------
    metrics : dict
        All evaluation metrics
    """
    all_metrics = {}
    
    if verbose:
        print("Running comprehensive evaluation...")
    
    # 1. Kinematic metrics
    if verbose:
        print("  [1/7] SAFD metrics...")
    all_metrics["safd_wasserstein"] = safd_wasserstein_distance(real_trips, synthetic_trips)
    
    if verbose:
        print("  [2/7] VSP metrics...")
    all_metrics.update(vsp_distribution_metrics(real_trips, synthetic_trips))
    
    if verbose:
        print("  [3/7] Jerk metrics...")
    all_metrics.update(jerk_metrics(real_trips, synthetic_trips))
    
    # 2. Statistical metrics
    if verbose:
        print("  [4/7] Distribution metrics...")
    all_metrics.update(speed_distribution_metrics(real_trips, synthetic_trips))
    all_metrics.update(acceleration_distribution_metrics(real_trips, synthetic_trips))
    all_metrics.update(duration_distribution_metrics(real_trips, synthetic_trips))
    all_metrics["mmd"] = compute_mmd(real_trips, synthetic_trips)
    
    # 3. Boundary metrics
    if verbose:
        print("  [5/7] Boundary metrics...")
    all_metrics.update(boundary_violation_rate(synthetic_trips))
    all_metrics.update(physical_constraint_violations(synthetic_trips))
    
    # 4. Utility metrics
    if verbose:
        print("  [6/7] Discriminative score...")
    all_metrics.update(discriminative_score(real_trips, synthetic_trips))
    
    if verbose:
        print("  [7/7] TSTR predictive score...")
    all_metrics.update(predictive_score_tstr(real_trips, synthetic_trips))
    
    if verbose:
        print("Evaluation complete!")
    
    return all_metrics


def create_evaluation_report(
    metrics: Dict[str, float],
    method_name: str = "PARSynthesizer"
) -> pd.DataFrame:
    """
    Create a formatted evaluation report DataFrame.
    """
    categories = {
        "Kinematic Fidelity": [
            "safd_wasserstein", "vsp_kl_divergence", "vsp_wasserstein",
            "ldlj_wasserstein", "jerk_max_mean_synth"
        ],
        "Statistical Fidelity": [
            "speed_wasserstein", "accel_wasserstein", "mmd",
            "speed_ks_statistic", "accel_ks_statistic"
        ],
        "Temporal Fidelity": [
            "tstr_mae", "tstr_ratio"
        ],
        "Boundary Conditions": [
            "boundary_any_violation_rate", "boundary_start_speed_max",
            "boundary_end_speed_max"
        ],
        "Physical Plausibility": [
            "phys_speed_violation_rate", "phys_accel_violation_rate",
            "phys_negative_speed_rate"
        ],
        "Utility": [
            "discriminative_score", "discriminative_auc"
        ],
    }
    
    rows = []
    for category, metric_names in categories.items():
        for name in metric_names:
            if name in metrics:
                value = metrics[name]
                rows.append({
                    "Category": category,
                    "Metric": name,
                    "Value": value,
                    "Method": method_name
                })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test with random data
    print("Testing evaluation metrics with synthetic example data...")
    
    np.random.seed(42)
    
    # Create fake "real" trips
    real_trips = []
    for _ in range(50):
        duration = np.random.randint(50, 200)
        t = np.linspace(0, np.pi, duration)
        speed = 15 * np.sin(t) + np.random.randn(duration) * 0.5
        speed = np.clip(speed, 0, None)
        speed[0] = 0
        speed[-1] = 0
        real_trips.append(speed)
    
    # Create fake "synthetic" trips (slightly different)
    synthetic_trips = []
    for _ in range(50):
        duration = np.random.randint(50, 200)
        t = np.linspace(0, np.pi, duration)
        speed = 14 * np.sin(t) + np.random.randn(duration) * 0.8
        speed = np.clip(speed, 0, None)
        speed[0] = 0
        speed[-1] = 0
        synthetic_trips.append(speed)
    
    # Run evaluation
    metrics = run_full_evaluation(real_trips, synthetic_trips, verbose=True)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


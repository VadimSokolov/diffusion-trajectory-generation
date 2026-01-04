import numpy as np
import scipy.sparse as sp

"""
Markov-bridge synthetic speed trajectory generator (1 Hz).

Key features:
- Learns a 2nd-order Markov model on discretized speed via pair-states (s_{t-1}, s_t).
- Samples trajectories with v(0)=0 and v(T)=0 exactly using a Markov bridge.
- Fast at scale: caches bridge backward messages (beta) per duration.
- Optional fast mean-speed control via *tilted bridge sampling* (no rejection sampling):
    P(x->y) ∝ P(x->y) * beta_{t+1}(y) * exp(lambda * speed(y))
  where lambda is selected (per duration) from a cached calibration grid.

Dependencies: numpy, scipy (sparse)

It is replacement for markov_bridge_speed_generator.py. It is faster and can generate many trips fast without rejection sampling.

### What changed

* **Caches the Markov-bridge backward messages** `beta` per duration (`N_seconds`) → you compute them once, then generate many trips fast.
* Adds **tilted bridge sampling** (`method="tilt"`) to hit a **target mean speed** without rejection:

  * It calibrates a small **λ grid** for each duration (cached), then samples once using
    [
    w \propto P(x\to y),\beta_{t+1}(y),e^{\lambda \cdot v(y)}
    ]
  * This is typically **orders of magnitude faster** than rejection when you need 10k–100k trips.

### Usage (same spirit as before)

```python
from markov_bridge_speed_generator_fast import fit_pair_markov, generate_trip, generate_trips

model = fit_pair_markov(trips, bin_w=0.5, alpha=0.01)

# Fast: duration only
v = generate_trip(model, N_seconds=240, method="none", seed=1)

# Fast: duration + mean speed targeting (NO rejection)
v = generate_trip(model, N_seconds=240, mean_speed_target=16.0, method="tilt", seed=2)

# Batch: 50k trips quickly (example)
durations = [240]*50000
means = [16.0]*50000
vs = generate_trips(model, n=50000, durations=durations, mean_speed_targets=means, method="tilt", seed=0)
```

If you later decide you want to condition on *distance* or *stop fraction*, the same tilting idea extends (tilt on other rewards), but mean speed is the high-impact one for eliminating rejection.


"""

# -----------------------------
# Utilities: binning & fitting
# -----------------------------

def make_speed_bins(trips, bin_w=0.5, v_max=None):
    vmax_data = max(float(np.max(v)) for v in trips)
    if v_max is None:
        v_max = max(5.0, vmax_data + 2 * bin_w)
    edges = np.arange(0.0, v_max + bin_w, bin_w)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers

def bin_speeds(v, edges):
    vmax = edges[-1] - 1e-9
    vv = np.clip(v, 0.0, vmax)
    s = np.digitize(vv, edges) - 1
    s = np.clip(s, 0, len(edges) - 2)
    return s.astype(np.int32)

def fit_pair_markov(trips, bin_w=0.5, v_max=None, alpha=0.01):
    """
    Fit a 2nd-order Markov model on speed bins using pair-states:
        X_t = (s_{t-1}, s_t)  in {0..K-1}^2  => S = K^2 states
    Valid transitions: (i,j) -> (j,k)
    """
    edges, centers = make_speed_bins(trips, bin_w=bin_w, v_max=v_max)
    K = len(centers)
    S = K * K

    row_idx, col_idx, data = [], [], []
    start_counts = np.zeros(S, dtype=np.float64)

    for v in trips:
        s = bin_speeds(v, edges)
        if len(s) < 2:
            continue
        pair = s[:-1] * K + s[1:]  # length N
        start_counts[pair[0]] += 1.0

        if len(pair) >= 2:
            rows = pair[:-1]
            cols = pair[1:]
            key = rows.astype(np.int64) * S + cols.astype(np.int64)
            uniq, cnt = np.unique(key, return_counts=True)
            row_idx.extend((uniq // S).astype(np.int32))
            col_idx.extend((uniq % S).astype(np.int32))
            data.extend(cnt.astype(np.float64))

    C = sp.coo_matrix((data, (row_idx, col_idx)), shape=(S, S)).tocsr()

    # Dirichlet smoothing on valid transitions only: (i,j)->(j,k)
    prior_rows, prior_cols, prior_data = [], [], []
    for i in range(K):
        for j in range(K):
            r = i * K + j
            ks = np.arange(K, dtype=np.int32)
            cs = j * K + ks
            prior_rows.append(np.full(K, r, dtype=np.int32))
            prior_cols.append(cs)
            prior_data.append(np.full(K, alpha, dtype=np.float64))
    prior_rows = np.concatenate(prior_rows)
    prior_cols = np.concatenate(prior_cols)
    prior_data = np.concatenate(prior_data)
    Prior = sp.coo_matrix((prior_data, (prior_rows, prior_cols)), shape=(S, S)).tocsr()

    C = C + Prior

    # Row-normalize => sparse transition matrix P
    row_sums = np.array(C.sum(axis=1)).ravel()
    inv = 1.0 / (row_sums + 1e-20)
    P = sp.diags(inv) @ C

    # Start distribution: first component must be 0-bin (because v(0)=0)
    mask_start = np.array([(idx // K) == 0 for idx in range(S)], dtype=bool)
    pi = start_counts + 1e-6
    pi[~mask_start] = 0.0
    if pi.sum() == 0:
        pi[mask_start] = 1.0
    pi = pi / pi.sum()

    # End states: any pair (i,0) so last speed bin is 0
    end_states = np.array([i * K + 0 for i in range(K)], dtype=np.int32)

    model = dict(
        P=P,
        edges=edges,
        centers=centers,  # bin centers in m/s
        K=K,
        S=S,
        pi=pi,
        end_states=end_states,
        bin_w=bin_w,
        # Caches for speed
        _beta_cache={},          # N_seconds -> beta array
        _tilt_cache={},          # (N_seconds, lambdas_tuple) -> dict with maps/arrays
        _lambda_map_cache={},    # N_seconds -> calibration dict: target->lambda via grid
    )
    return model

# -----------------------------
# Bridge cache
# -----------------------------

def precompute_beta(model, N_seconds):
    """
    Compute and cache Markov-bridge backward messages beta[t, x] ∝ P(reach end at time N | X_t=x).
    Normalized per t for numeric stability.
    """
    cache = model["_beta_cache"]
    if N_seconds in cache:
        return cache[N_seconds]

    P = model["P"]
    S = model["S"]
    end_states = model["end_states"]
    N = int(N_seconds)

    beta = np.zeros((N + 1, S), dtype=np.float32)
    beta[N, end_states] = 1.0
    beta[N] /= (beta[N].sum() + 1e-20)

    # Backward recursion
    for t in range(N - 1, 0, -1):
        bt = (P @ beta[t + 1]).astype(np.float32)
        s = float(bt.sum())
        if s > 0:
            bt /= s
        beta[t] = bt

    cache[N_seconds] = beta
    return beta

# -----------------------------
# Sampling
# -----------------------------

def _categorical_choice(rng, cols, w):
    """
    Robust categorical sampling with unnormalized, possibly imperfect weights.
    cols: array of candidate states (int)
    w: weights (float)
    """
    w = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return cols[int(rng.integers(0, len(cols)))]
    w = w / s
    cdf = np.cumsum(w)
    u = rng.random()
    j = int(np.searchsorted(cdf, u, side="right"))
    if j >= len(cols):
        j = len(cols) - 1
    return cols[j]

def sample_bridge(model, N_seconds, seed=0, inertia_lambda=0.7):
    """
    Unconditional bridge sampling (no mean targeting), with optional inertia penalty.
    Returns v[0..N] (length N+1) in m/s with endpoints 0.
    """
    rng = np.random.default_rng(seed)
    P = model["P"]
    K = model["K"]
    S = model["S"]
    pi = model["pi"]
    centers = model["centers"]
    beta = precompute_beta(model, N_seconds)
    N = int(N_seconds)

    # Start state
    w1 = pi * beta[1]
    x = _categorical_choice(rng, np.arange(S, dtype=np.int32), w1)

    states = np.empty(N + 1, dtype=np.int32)
    states[1] = x

    for t in range(1, N):
        x = states[t]
        sptr, eptr = P.indptr[x], P.indptr[x + 1]
        cols = P.indices[sptr:eptr].astype(np.int32)
        probs = P.data[sptr:eptr].astype(np.float64)

        w = probs * beta[t + 1, cols].astype(np.float64)

        if inertia_lambda and inertia_lambda > 0:
            j = x % K
            k = cols % K
            delta = (k - j).astype(np.float64)
            w = w * np.exp(-inertia_lambda * (delta ** 2))

        states[t + 1] = _categorical_choice(rng, cols, w)

    # Reconstruct speeds from pair-states
    s = np.empty(N + 1, dtype=np.int32)
    s[0] = 0
    for t in range(1, N + 1):
        s[t] = states[t] % K

    v = centers[s].astype(np.float64)
    v[0] = 0.0
    v[-1] = 0.0
    return v

def _tilt_exp_cache(model, lambdas):
    """
    Cache exp(lambda * speed_center) per lambda for fast sampling.
    """
    centers = model["centers"]
    K = model["K"]
    lambdas = tuple(float(x) for x in lambdas)
    key = ("exp_speed", lambdas)
    tc = model["_tilt_cache"]
    if key in tc:
        return tc[key]
    exp_speed = np.stack([np.exp(lmb * centers).astype(np.float64) for lmb in lambdas], axis=0)  # (L,K)
    tc[key] = exp_speed
    return exp_speed

def sample_bridge_tilted(model, N_seconds, lam, seed=0, inertia_lambda=0.0):
    """
    Bridge sampling with exponential tilting to control expected mean speed (distance).
    Weights:
      w ∝ P(x->y) * beta_{t+1}(y) * exp(lam * speed(next_bin))
    inertia_lambda optional (can be 0 if tilt is used).
    """
    rng = np.random.default_rng(seed)
    P = model["P"]
    K = model["K"]
    S = model["S"]
    pi = model["pi"]
    centers = model["centers"]
    beta = precompute_beta(model, N_seconds)
    N = int(N_seconds)

    # Start state
    w1 = pi * beta[1]
    x = _categorical_choice(rng, np.arange(S, dtype=np.int32), w1)

    states = np.empty(N + 1, dtype=np.int32)
    states[1] = x

    for t in range(1, N):
        x = states[t]
        sptr, eptr = P.indptr[x], P.indptr[x + 1]
        cols = P.indices[sptr:eptr].astype(np.int32)
        probs = P.data[sptr:eptr].astype(np.float64)

        # next speed bin index
        k = cols % K
        w = probs * beta[t + 1, cols].astype(np.float64) * np.exp(lam * centers[k])

        if inertia_lambda and inertia_lambda > 0:
            j = x % K
            delta = (k - j).astype(np.float64)
            w = w * np.exp(-inertia_lambda * (delta ** 2))

        states[t + 1] = _categorical_choice(rng, cols, w)

    s = np.empty(N + 1, dtype=np.int32)
    s[0] = 0
    for t in range(1, N + 1):
        s[t] = states[t] % K

    v = centers[s].astype(np.float64)
    v[0] = 0.0
    v[-1] = 0.0
    return v

# -----------------------------
# Postprocessing (optional)
# -----------------------------

def smooth_and_project(v, a_max=2.8, window=7):
    """
    Light postprocessing:
    - smooth acceleration with moving average
    - enforce sum(a)=0 (end at 0)
    - cap accel magnitude
    - clip negatives
    """
    v = np.asarray(v, dtype=np.float64).copy()
    N = len(v) - 1
    if N <= 2:
        v[0] = 0.0
        v[-1] = 0.0
        return v

    a = np.diff(v)
    w = max(3, int(window) | 1)  # odd
    kernel = np.ones(w) / w
    a_s = np.convolve(a, kernel, mode="same")

    # enforce end-at-zero: sum a must be 0
    drift = a_s.sum()
    a_s = a_s - drift / max(N, 1)

    a_s = np.clip(a_s, -a_max, a_max)

    v2 = np.concatenate([[0.0], np.cumsum(a_s)])
    v2[-1] = 0.0
    v2 = np.clip(v2, 0.0, None)
    v2[0] = 0.0
    v2[-1] = 0.0
    return v2

def _apply_mean_scale(v, mean_speed_target, v_max=None):
    v = np.asarray(v, dtype=np.float64).copy()
    target = float(mean_speed_target)
    cur = float(v.mean())
    if cur > 1e-12:
        v = v * (target / cur)
    v[0] = 0.0
    v[-1] = 0.0
    if v_max is not None:
        v = np.clip(v, 0.0, float(v_max))
        v[0] = 0.0
        v[-1] = 0.0
    return v

# -----------------------------
# Fast mean-speed targeting via lambda calibration (no rejection)
# -----------------------------

def calibrate_lambda_grid(model, N_seconds, lambdas=None, mc_per_lambda=40, seed=0, inertia_lambda=0.0):
    """
    Build (and cache) a mapping: lambda -> expected mean speed for a fixed duration N_seconds,
    estimated by Monte Carlo sampling.

    This is used to pick lambda for a requested mean speed without rejection sampling.

    Notes:
    - This calibration is done per duration. You can optionally bucket durations yourself.
    - mc_per_lambda can be small (20–50) because we only need a coarse monotone map.
    """
    N = int(N_seconds)
    if lambdas is None:
        # Conservative default grid. You can adjust later after seeing ranges.
        lambdas = np.linspace(-0.20, 0.20, 21)

    lambdas = tuple(float(x) for x in lambdas)
    cache_key = (N, lambdas, int(mc_per_lambda), float(inertia_lambda))
    lm_cache = model["_lambda_map_cache"]
    if cache_key in lm_cache:
        return lm_cache[cache_key]

    rng = np.random.default_rng(seed)
    means = []
    for i, lam in enumerate(lambdas):
        # vary seed per lambda for reproducibility
        m = 0.0
        for j in range(int(mc_per_lambda)):
            v = sample_bridge_tilted(model, N, lam, seed=int(rng.integers(0, 2**31-1)), inertia_lambda=inertia_lambda)
            m += float(v.mean())
        means.append(m / float(mc_per_lambda))

    means = np.asarray(means, dtype=np.float64)
    out = {"lambdas": np.asarray(lambdas, dtype=np.float64), "mean_speeds": means}
    lm_cache[cache_key] = out
    return out

def pick_lambda_for_mean(calib, mean_speed_target):
    """
    Pick lambda via linear interpolation on the calibrated lambda->mean map.
    """
    target = float(mean_speed_target)
    lambdas = calib["lambdas"]
    means = calib["mean_speeds"]

    # If not monotone due to MC noise, sort by means
    order = np.argsort(means)
    means_s = means[order]
    lambdas_s = lambdas[order]

    if target <= means_s[0]:
        return float(lambdas_s[0])
    if target >= means_s[-1]:
        return float(lambdas_s[-1])

    idx = np.searchsorted(means_s, target, side="left")
    lo = idx - 1
    hi = idx
    # linear interpolation
    m0, m1 = means_s[lo], means_s[hi]
    l0, l1 = lambdas_s[lo], lambdas_s[hi]
    if abs(m1 - m0) < 1e-12:
        return float(l0)
    w = (target - m0) / (m1 - m0)
    return float(l0 + w * (l1 - l0))

# -----------------------------
# Public API: single trip + batch
# -----------------------------

def generate_trip(
    model,
    N_seconds,
    mean_speed_target=None,
    method="tilt",
    tol=0.25,
    tries=300,
    seed=0,
    inertia_lambda=0.7,
    postprocess=True,
    a_max=2.8,
    window=7,
    v_max=None,
    # calibration knobs for tilt
    lambdas=None,
    mc_per_lambda=40,
    calib_seed=1234,
):
    """
    Generate one synthetic trip of duration N_seconds (integer), returning v[0..N] in m/s.

    mean_speed_target:
      - if None: unconditional bridge
      - if set: try to hit target mean speed

    method:
      - "tilt": fast mean targeting via lambda calibration and tilted sampling (recommended at scale)
      - "reject": rejection sampling on mean speed (kept for reference)
      - "none": unconditional bridge

    postprocess:
      - if True: smooth/project, then optionally rescale to mean_speed_target
    """
    N = int(N_seconds)

    if mean_speed_target is None or method == "none":
        v = sample_bridge(model, N, seed=seed, inertia_lambda=inertia_lambda)
    elif method == "reject":
        rng = np.random.default_rng(seed)
        best = None
        best_err = np.inf
        target = float(mean_speed_target)
        for _ in range(int(tries)):
            v0 = sample_bridge(model, N, seed=int(rng.integers(0, 2**31-1)), inertia_lambda=inertia_lambda)
            err = abs(float(v0.mean()) - target)
            if err < best_err:
                best, best_err = v0, err
            if err <= float(tol):
                best = v0
                break
        v = best
    else:
        # Tilted bridge: calibrate lambda->mean map once per N (cached), then sample once
        calib = calibrate_lambda_grid(
            model, N, lambdas=lambdas, mc_per_lambda=mc_per_lambda, seed=calib_seed, inertia_lambda=0.0
        )
        lam = pick_lambda_for_mean(calib, mean_speed_target)
        v = sample_bridge_tilted(model, N, lam, seed=seed, inertia_lambda=0.0)

    if postprocess:
        v = smooth_and_project(v, a_max=a_max, window=window)
        if mean_speed_target is not None:
            v = _apply_mean_scale(v, mean_speed_target, v_max=v_max)
            # one more light projection to reduce small endpoint/clip artifacts
            v = smooth_and_project(v, a_max=a_max, window=window)
            if v_max is not None:
                v = np.clip(v, 0.0, float(v_max))
                v[0] = 0.0
                v[-1] = 0.0

    return v

def generate_trips(
    model,
    n,
    durations,
    mean_speed_targets=None,
    method="tilt",
    seed=0,
    **kwargs
):
    """
    Batch generation helper.

    durations:
      - either an int (fixed duration for all),
      - or an array/list of length n with durations per trip.

    mean_speed_targets:
      - None (unconditional), or float, or array/list length n.

    Returns: list of numpy arrays (each length N_seconds+1).
    """
    rng = np.random.default_rng(seed)

    if isinstance(durations, (int, np.integer)):
        durations = [int(durations)] * int(n)
    else:
        durations = list(durations)
        if len(durations) != int(n):
            raise ValueError("durations must be an int or a list/array of length n")

    if mean_speed_targets is None:
        mean_speed_targets = [None] * int(n)
    elif isinstance(mean_speed_targets, (float, int, np.floating, np.integer)):
        mean_speed_targets = [float(mean_speed_targets)] * int(n)
    else:
        mean_speed_targets = list(mean_speed_targets)
        if len(mean_speed_targets) != int(n):
            raise ValueError("mean_speed_targets must be None, a scalar, or a list/array of length n")

    out = []
    for i in range(int(n)):
        N = int(durations[i])
        tgt = mean_speed_targets[i]
        trip_seed = int(rng.integers(0, 2**31-1))
        v = generate_trip(
            model,
            N_seconds=N,
            mean_speed_target=tgt,
            method=method,
            seed=trip_seed,
            **kwargs
        )
        out.append(v)
    return out

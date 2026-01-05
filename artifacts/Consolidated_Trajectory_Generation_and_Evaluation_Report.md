# Consolidated Report: Synthetic Vehicle Speed Trajectory Generation & Evaluation
**Addressing Mode Collapse, Enhancing Fidelity, and Ensuring Physical Validity**

**Date:** 2025-12-31

---

## 1. Executive Summary & Diagnostic
The objective is to generate synthetic vehicle speed "micro-trips" (1 Hz resolution) that **start and end at 0 m/s**, with strict control over **duration** and **average speed**.

*   **Current Status:** "Limited success" with GANs; **Mode Collapse** with `PARSynthesizer` (SDV).
*   **Diagnosis of Failures:**
    *   **PARSynthesizer (Mode Collapse):** PAR models suffer from exposure bias. Without high "temperature" sampling or rich conditioning, they revert to the "mean" behavior.
    *   **GANs (Limited Success):** Standard TimeGANs struggle with **hard constraints** ($v_{start}=v_{end}=0$) and often produce "jittery" physics.
*   **Strategic Pivot:** The recommendation is to move towards **Denoising Diffusion Probabilistic Models (DDPMs)** for their inherent "inpainting" capabilities which solve the boundary constraint problem, or **Quantized Transformers** for better diversity control.

---

## 2. Data Engineering & Segmentation Strategy

Quality synthetic generation begins with rigorous data segmentation. Raw telematics data must be processed into discrete "micro-trips" to serve as valid training examples.

### 2.1 Micro-Trip Definition
A "micro-trip" is defined as a velocity profile between two successive dwell (idling) events. It must start at $v=0$, accelerate to a speed $> \epsilon$, and decelerate back to $v=0$.

### 2.2 Segmentation Algorithm
1.  **Input:** Continuous time-series array $V = [v_0, v_1, \dots, v_N]$.
2.  **Dwell Identification:** Identify indices $I = \{i \mid v_i < \epsilon\}$, where $\epsilon$ is a near-zero threshold (e.g., 0.1 m/s).
3.  **Segmentation:** For each pair of consecutive indices $(i, j)$ in $I$ where $j > i + 1$:
    *   Extract segment $S_k = V[i:j+1]$.
    *   **Validation:** Ensure $\max(S_k) > 2$ m/s to filter out GPS drift/noise.
4.  **Feature Extraction (Conditioning):**
    *   **Duration ($D_k$):** Length of segment.
    *   **Average Speed ($\bar{v}_k$):** Mean valid speed.
    *   **Traffic Ratio:** $R = \text{spd\_traffic} / \text{spd\_base}$ (if available) to infer congestion.

---

## 3. Modeling Architectures: From Stochastic to SOTA

### 3.1 Baseline: Markov Chains (The "Transparent" Approach)
*   **Mechanism:** State space defined by $(v, a)$. Transition Probability Matrices (TPM) $P_{ij}$ learned from data.
*   **Controlling Duration/Speed:**
    *   **Clustering:** Train separate TPMs for "High Speed" vs. "Low Speed" trips.
    *   **Rejection Sampling:** Generate thousands of random walks; discard those that don't match target $T$ or $\bar{v}$ (Cost is negligible).
*   **Pros:** Physically explainable, easy to implement in R/Python.
*   **Cons:** "Memoryless" (lacks long-term planning), high-frequency jitter.

### 3.2 Deep Generative Models (DGMs)

#### A. Conditional TimeGAN (cTimeGAN)
*   **Mechanism:** Adversarial learning ($G$ vs. $D$) with a stepwise supervised loss.
*   **Control:** Target Duration and Avg Speed are injected as **Static Features** into the latent embedding.
*   **Fixing Failures:** Requires **DoppelGANger** architecture (Lin et al.) to separate Attribute Generation from Feature Generation, reducing mode collapse.

#### B. PARSynthesizer (SDV)
*   **Mechanism:** Probabilistic Auto-Regressive model.
*   **Control:** Define `TripID` as sequence key and targets as `context_columns`.
*   **Fixing Mode Collapse:**
    *   **Context Noise:** Add slight Gaussian noise to context variables during sampling.
    *   **Temperature:** Increase sampling temperature ($T > 1.0$) to encourage rare transitions.

### 3.3 The "Silver Bullet": Diffusion Models & Inpainting
*   **Mechanism:** Learn the data distribution's gradients (score matching) to denoise random Gaussian noise into trajectories.
*   **Key Advantage (Inpainting):** Can strictly enforce boundary conditions.
    *   **Method:** During the reverse process $x_t \to x_{t-1}$, fix $x_{start}$ and $x_{end}$ to 0 at every step. The model "fills in" the trajectory between these anchors.
*   **Libraries:** `tsgm`, `diffusers`, or `TimeWeaver` (SOTA for conditional generation).

### 3.4 Foundation Models: Amazon Chronos
*   **Mechanism:** Pre-trained probabilistic time-series model (T5 architecture).
*   **Usage:** Zero-shot or fine-tuned generation. Provide start token "0" and forecast $N$ steps.
*   **Pros:** Learned universal temporal dynamics; less training data needed.

---

## 4. Comprehensive Evaluation Framework

To prove validity, meaningfulness, and safety, evaluation must go beyond simple accuracy.

### 4.1 Physics & Kinematics (Validity)
*   **Speed-Acceleration Frequency Distribution (SAFD):** 2D Histogram of $(v, a)$.
    *   *Check:* Coverage of "launch" (high $a$, low $v$) and "braking" tails. Use **Earth Mover's Distance (EMD)**.
*   **Vehicle Specific Power (VSP):**
    *   *Metric:* $VSP = v \cdot (1.1a + 0.132) + 0.000302v^3$.
    *   *Check:* GANs often produce "fat tails" here due to jitter.
*   **Log Dimensionless Jerk (LDLJ):**
    *   *Metric:* Sensitivity to smoothness/comfort. High LDLJ = unnatural "jitter" (common in TimeGAN). Low LDLJ = over-smoothed (common in VAEs).

### 4.2 Temporal & Distributional Fidelity
*   **Dynamic Time Warping (DTW):** Measures shape similarity ignoring phase shifts.
*   **Fréchet Distance:** The "dog-walking" distance. Measures worst-case deviation; critical for safety limits.
*   **Wasserstein Distance ($W_1$):** Best for checking if the synthetic distribution "covers" the real distribution.
*   **Maximum Mean Discrepancy (MMD):** Kernel-based test for distributional equality.

### 4.3 Diversity & Privacy
*   **Mode Collapse Checks:**
    *   **Precision (Quality):** Do synthetic trips look real?
    *   **Recall (Diversity):** Are ALL real behaviors represented? (High Recall = No Mode Collapse).
*   **Privacy - Exact Match Score:**
    *   Distance between synthetic trip and nearest training neighbor. If 0, the model has memorized data (Privacy Breach).

### 4.4 Conditioning Check (Utility)
*   **Joint Fréchet Time Series Distance (J-FTSD):**
    *   Measures distance between joint distributions $(Trajectory, Condition)_{real}$ vs $(Trajectory, Condition)_{syn}$. Ensures generated trip matches the requested "Context".

---

## 5. Implementation Recommendations

**Rank 1: Diffusion (TimeWeaver/TSDiff)**
*   **Why:** Solves the hard constraint ($v=0$) problem mathematically via inpainting. Best diversity (Recall).
*   **Action:** Implement using `tsgm` or custom PyTorch loop.

**Rank 2: Quantized Transformer (GPT)**
*   **Why:** Explicit "Temperature" knob to force diversity and fix mode collapse.
*   **Action:** Discretize speed (0-120) and train `minGPT` with temperature sampling.

**Rank 3: Improved PAR/GAN (SDV/DoppelGANger)**
*   **Why:** Easier APIs (`sdv`, `ydata`).
*   **Action:** Only if Diffusion is too computationally expensive. MUST use "Context Noise" or "DoppelGANger" architecture to fight mode collapse.

**Rank 4: Markov Chains (R/Python)**
*   **Why:** Robust baseline.
*   **Action:** Use if deep learning fails. Use Cluster-then-Generate strategy.

### Next Steps
1.  **Step 1:** Run segmentation algorithm on source data to create `micro_trips.csv`.
2.  **Step 2:** Choose **Diffusion** (for quality) or **Transformer** (for control).
3.  **Step 3:** Implement the **Inpainting** sampling loop.
4.  **Step 4:** Evaluate using **SAFD (Physics)** and **Recall (Diversity)**.

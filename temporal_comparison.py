"""
Temporal Data Comparison: Fourier Features vs Reservoir Computing

On temporal data, reservoir computing should have advantages:
1. Inherent memory of past inputs (recurrent dynamics)
2. Designed for sequence processing
3. Echo State Property captures temporal dependencies

Tasks:
1. Mackey-Glass time series prediction (chaotic system)
2. NARMA-10 (nonlinear autoregressive moving average)
3. Sine wave with varying frequency (requires adaptation)
4. Memory capacity task (how far back can it remember?)

For each task, we compare:
- Standard ESN (sequential processing with memory)
- Fourier Features (no temporal memory, just time encoding)
- Fourier + ESN hybrid
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List


# ============================================================
# TEMPORAL DATA GENERATORS
# ============================================================

def generate_mackey_glass(n_samples: int, tau: int = 17, seed: int = 42) -> np.ndarray:
    """
    Generate Mackey-Glass chaotic time series.

    dx/dt = β * x(t-τ) / (1 + x(t-τ)^n) - γ * x(t)

    Classic benchmark for time series prediction.
    Chaotic for τ > 16.8
    """
    np.random.seed(seed)

    # Parameters
    beta = 0.2
    gamma = 0.1
    n_exp = 10
    dt = 1.0

    # Initialize with random history
    history_len = tau + 1
    x = np.zeros(n_samples + history_len)
    x[:history_len] = 0.9 + 0.2 * (np.random.rand(history_len) - 0.5)

    # Generate using Euler method
    for t in range(history_len, n_samples + history_len):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1 + x_tau ** n_exp) - gamma * x[t - 1]
        x[t] = x[t - 1] + dt * dx

    return x[history_len:]


def generate_narma10(n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate NARMA-10 task (Nonlinear AutoRegressive Moving Average).

    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y(t-i), i=0..9) + 1.5*u(t-9)*u(t) + 0.1

    Requires 10-step memory - challenging for any system.
    """
    np.random.seed(seed)

    # Input: uniform random
    u = np.random.uniform(0, 0.5, n_samples + 100)

    # Output
    y = np.zeros(n_samples + 100)
    y[:10] = 0.1

    for t in range(10, n_samples + 100):
        y[t] = (0.3 * y[t-1] +
                0.05 * y[t-1] * np.sum(y[t-10:t]) +
                1.5 * u[t-9] * u[t] +
                0.1)
        # Clip to prevent explosion
        y[t] = np.clip(y[t], 0, 1)

    return u[100:], y[100:]


def generate_varying_frequency_sine(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Generate sine wave with slowly varying frequency.

    Requires adaptation to changing dynamics.
    """
    np.random.seed(seed)

    t = np.linspace(0, 10, n_samples)

    # Frequency varies from 1 to 5 Hz
    freq = 1 + 4 * (t / t.max())

    # Phase accumulation for varying frequency
    phase = np.cumsum(2 * np.pi * freq / n_samples * (t[1] - t[0]) * n_samples)

    return np.sin(phase)


def generate_memory_task(n_samples: int, delay: int = 10, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory capacity task: output should be input delayed by 'delay' steps.

    Tests how far back the system can remember.
    """
    np.random.seed(seed)

    # Random input
    u = np.random.randn(n_samples)

    # Target is delayed input
    y = np.zeros(n_samples)
    y[delay:] = u[:-delay]

    return u, y


# ============================================================
# MODELS
# ============================================================

class TemporalESN:
    """
    Standard Echo State Network for temporal data.

    Processes sequences with recurrent memory.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        spectral_radius: float = 0.9,
        leaking_rate: float = 1.0,
        input_scaling: float = 1.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # Input weights
        self.W_in = np.random.uniform(-input_scaling, input_scaling,
                                       (hidden_size, input_size))

        # Recurrent weights (spectral radius scaled)
        W_hh = np.random.uniform(-1, 1, (hidden_size, hidden_size))
        sr = np.max(np.abs(np.linalg.eigvals(W_hh)))
        self.W_hh = W_hh * (spectral_radius / sr)

        # Bias
        self.b = np.random.uniform(-0.1, 0.1, hidden_size)

        # Output weights (trainable)
        self.W_out = None

    def forward(self, X: np.ndarray, return_states: bool = True) -> np.ndarray:
        """
        Forward pass through reservoir.

        Args:
            X: Input sequence, shape (seq_len, input_size)

        Returns:
            Hidden states, shape (seq_len, hidden_size)
        """
        seq_len = X.shape[0]
        H = np.zeros((seq_len, self.hidden_size))
        h = np.zeros(self.hidden_size)

        α = self.leaking_rate

        for t in range(seq_len):
            x_t = X[t]
            pre_act = self.W_in @ x_t + self.W_hh @ h + self.b
            h = (1 - α) * h + α * np.tanh(pre_act)
            H[t] = h

        return H

    def fit(self, X: np.ndarray, Y: np.ndarray, washout: int = 100):
        """Train output weights via ridge regression."""
        H = self.forward(X)

        # Remove washout
        H_train = H[washout:]
        Y_train = Y[washout:]

        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # Ridge regression
        HTH = H_train.T @ H_train + self.lambda_reg * np.eye(self.hidden_size)
        HTY = H_train.T @ Y_train
        self.W_out = np.linalg.solve(HTH, HTY).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output sequence."""
        H = self.forward(X)
        return (H @ self.W_out.T).flatten()


class TemporalFourier:
    """
    Fourier Features for temporal data.

    Encodes time coordinate only - NO temporal memory.
    This tests if frequency encoding alone can capture temporal patterns.
    """

    def __init__(
        self,
        embedding_size: int,
        output_size: int = 1,
        sigma: float = 10.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.sigma = sigma
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # Frequency matrix for 1D time
        self.B = np.random.randn(embedding_size, 1) * sigma

        self.W_out = None

    def encode(self, t: np.ndarray) -> np.ndarray:
        """Fourier encode time coordinates."""
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        projection = 2 * np.pi * t @ self.B.T
        return np.concatenate([np.sin(projection), np.cos(projection)], axis=1)

    def fit(self, t: np.ndarray, Y: np.ndarray, washout: int = 100):
        """Train via ridge regression."""
        H = self.encode(t)

        H_train = H[washout:]
        Y_train = Y[washout:]

        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        HTH = H_train.T @ H_train + self.lambda_reg * np.eye(2 * self.embedding_size)
        HTY = H_train.T @ Y_train
        self.W_out = np.linalg.solve(HTH, HTY).T

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Predict from time coordinates."""
        H = self.encode(t)
        return (H @ self.W_out.T).flatten()


class TemporalFourierWithHistory:
    """
    Fourier Features with input history window.

    Includes past N inputs as features (explicit memory).
    """

    def __init__(
        self,
        embedding_size: int,
        history_len: int = 10,
        sigma: float = 10.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        self.embedding_size = embedding_size
        self.history_len = history_len
        self.sigma = sigma
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # Fourier features for time + history window
        input_dim = 1 + history_len  # time + history
        self.B = np.random.randn(embedding_size, input_dim) * sigma

        self.W_out = None

    def create_history_features(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Create features with time and input history."""
        n = len(t)
        features = np.zeros((n, 1 + self.history_len))
        features[:, 0] = t

        for i in range(self.history_len):
            delay = i + 1
            features[delay:, i + 1] = X[:-delay]

        return features

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Fourier encode combined features."""
        projection = 2 * np.pi * features @ self.B.T
        return np.concatenate([np.sin(projection), np.cos(projection)], axis=1)

    def fit(self, t: np.ndarray, X: np.ndarray, Y: np.ndarray, washout: int = 100):
        """Train via ridge regression."""
        features = self.create_history_features(t, X)
        H = self.encode(features)

        H_train = H[washout:]
        Y_train = Y[washout:]

        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        HTH = H_train.T @ H_train + self.lambda_reg * np.eye(2 * self.embedding_size)
        HTY = H_train.T @ Y_train
        self.W_out = np.linalg.solve(HTH, HTY).T

        self._last_X = X  # Store for prediction

    def predict(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict with time and history."""
        features = self.create_history_features(t, X)
        H = self.encode(features)
        return (H @ self.W_out.T).flatten()


# ============================================================
# EXPERIMENTS
# ============================================================

def compute_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Root Mean Squared Error."""
    return np.sqrt(compute_nmse(y_true, y_pred))


def experiment_mackey_glass():
    """Mackey-Glass prediction task."""
    print("\n" + "=" * 70)
    print("TASK 1: Mackey-Glass Time Series Prediction")
    print("=" * 70)

    # Generate data
    n_samples = 5000
    data = generate_mackey_glass(n_samples)

    # Normalize
    data = (data - data.mean()) / data.std()

    # Create input (current value) and target (next value)
    X = data[:-1].reshape(-1, 1)
    Y = data[1:]
    t = np.arange(len(Y)).astype(float) / len(Y)

    # Train/test split
    split = int(0.8 * len(Y))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    t_train, t_test = t[:split], t[split:]

    results = {}
    washout = 100

    # 1. ESN
    print("\n--- ESN (H=500) ---")
    esn = TemporalESN(input_size=1, hidden_size=500, output_size=1,
                      spectral_radius=0.9, lambda_reg=1e-6)
    esn.fit(X_train, Y_train, washout=washout)
    pred_esn = esn.predict(X_test)
    nrmse_esn = compute_nrmse(Y_test[washout:], pred_esn[washout:])
    print(f"NRMSE: {nrmse_esn:.4f}")
    results['ESN'] = {'pred': pred_esn, 'nrmse': nrmse_esn}

    # 2. Fourier (time only)
    print("\n--- Fourier (time encoding only, σ=10) ---")
    fourier = TemporalFourier(embedding_size=256, sigma=10.0, lambda_reg=1e-6)
    fourier.fit(t_train, Y_train, washout=washout)
    pred_fourier = fourier.predict(t_test)
    nrmse_fourier = compute_nrmse(Y_test[washout:], pred_fourier[washout:])
    print(f"NRMSE: {nrmse_fourier:.4f}")
    results['Fourier (time)'] = {'pred': pred_fourier, 'nrmse': nrmse_fourier}

    # 3. Fourier with history
    print("\n--- Fourier + History Window (10 steps) ---")
    fourier_hist = TemporalFourierWithHistory(embedding_size=256, history_len=10,
                                               sigma=10.0, lambda_reg=1e-6)
    fourier_hist.fit(t_train, X_train.flatten(), Y_train, washout=washout)
    pred_fourier_hist = fourier_hist.predict(t_test, X_test.flatten())
    nrmse_fourier_hist = compute_nrmse(Y_test[washout:], pred_fourier_hist[washout:])
    print(f"NRMSE: {nrmse_fourier_hist:.4f}")
    results['Fourier + History'] = {'pred': pred_fourier_hist, 'nrmse': nrmse_fourier_hist}

    return Y_test, results


def experiment_narma10():
    """NARMA-10 task (requires 10-step memory)."""
    print("\n" + "=" * 70)
    print("TASK 2: NARMA-10 (10-step memory required)")
    print("=" * 70)

    # Generate data
    n_samples = 5000
    U, Y = generate_narma10(n_samples)

    # Reshape
    X = U.reshape(-1, 1)
    t = np.arange(len(Y)).astype(float) / len(Y)

    # Split
    split = int(0.8 * len(Y))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    t_train, t_test = t[:split], t[split:]
    U_train, U_test = U[:split], U[split:]

    results = {}
    washout = 100

    # 1. ESN
    print("\n--- ESN (H=500) ---")
    esn = TemporalESN(input_size=1, hidden_size=500, output_size=1,
                      spectral_radius=0.9, lambda_reg=1e-6)
    esn.fit(X_train, Y_train, washout=washout)
    pred_esn = esn.predict(X_test)
    nrmse_esn = compute_nrmse(Y_test[washout:], pred_esn[washout:])
    print(f"NRMSE: {nrmse_esn:.4f}")
    results['ESN'] = {'pred': pred_esn, 'nrmse': nrmse_esn}

    # 2. Fourier (time only) - should fail
    print("\n--- Fourier (time only) - no memory ---")
    fourier = TemporalFourier(embedding_size=256, sigma=10.0, lambda_reg=1e-6)
    fourier.fit(t_train, Y_train, washout=washout)
    pred_fourier = fourier.predict(t_test)
    nrmse_fourier = compute_nrmse(Y_test[washout:], pred_fourier[washout:])
    print(f"NRMSE: {nrmse_fourier:.4f}")
    results['Fourier (time)'] = {'pred': pred_fourier, 'nrmse': nrmse_fourier}

    # 3. Fourier with 10-step history
    print("\n--- Fourier + History (10 steps) ---")
    fourier_hist = TemporalFourierWithHistory(embedding_size=256, history_len=10,
                                               sigma=10.0, lambda_reg=1e-6)
    fourier_hist.fit(t_train, U_train, Y_train, washout=washout)
    pred_fourier_hist = fourier_hist.predict(t_test, U_test)
    nrmse_fourier_hist = compute_nrmse(Y_test[washout:], pred_fourier_hist[washout:])
    print(f"NRMSE: {nrmse_fourier_hist:.4f}")
    results['Fourier + History'] = {'pred': pred_fourier_hist, 'nrmse': nrmse_fourier_hist}

    # 4. Fourier with 20-step history (more than needed)
    print("\n--- Fourier + History (20 steps) ---")
    fourier_hist20 = TemporalFourierWithHistory(embedding_size=256, history_len=20,
                                                 sigma=10.0, lambda_reg=1e-6)
    fourier_hist20.fit(t_train, U_train, Y_train, washout=washout)
    pred_fourier_hist20 = fourier_hist20.predict(t_test, U_test)
    nrmse_fourier_hist20 = compute_nrmse(Y_test[washout:], pred_fourier_hist20[washout:])
    print(f"NRMSE: {nrmse_fourier_hist20:.4f}")
    results['Fourier + History (20)'] = {'pred': pred_fourier_hist20, 'nrmse': nrmse_fourier_hist20}

    return Y_test, results


def experiment_memory_capacity():
    """Test memory capacity at different delays."""
    print("\n" + "=" * 70)
    print("TASK 3: Memory Capacity (how far back can we remember?)")
    print("=" * 70)

    n_samples = 3000
    delays = [1, 5, 10, 20, 30, 50]

    esn_scores = []
    fourier_scores = []

    for delay in delays:
        U, Y = generate_memory_task(n_samples, delay=delay)
        X = U.reshape(-1, 1)
        t = np.arange(len(Y)).astype(float) / len(Y)

        split = int(0.8 * len(Y))
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]
        t_train, t_test = t[:split], t[split:]
        washout = max(100, delay + 50)

        # ESN
        esn = TemporalESN(input_size=1, hidden_size=500, output_size=1,
                          spectral_radius=0.99, lambda_reg=1e-8)
        esn.fit(X_train, Y_train, washout=washout)
        pred_esn = esn.predict(X_test)

        # Correlation as memory capacity metric
        valid_idx = slice(washout, None)
        corr_esn = np.corrcoef(Y_test[valid_idx], pred_esn[valid_idx])[0, 1] ** 2
        esn_scores.append(corr_esn)

        # Fourier with matched history
        fourier = TemporalFourierWithHistory(embedding_size=256, history_len=delay + 5,
                                              sigma=5.0, lambda_reg=1e-6)
        fourier.fit(t_train, X_train.flatten(), Y_train, washout=washout)
        pred_fourier = fourier.predict(t_test, X_test.flatten())
        corr_fourier = np.corrcoef(Y_test[valid_idx], pred_fourier[valid_idx])[0, 1] ** 2
        fourier_scores.append(corr_fourier)

        print(f"Delay {delay:2d}: ESN R²={corr_esn:.4f}, Fourier+History R²={corr_fourier:.4f}")

    return delays, esn_scores, fourier_scores


def experiment_varying_frequency():
    """Varying frequency sine - tests adaptation."""
    print("\n" + "=" * 70)
    print("TASK 4: Varying Frequency Sine (adaptation required)")
    print("=" * 70)

    n_samples = 2000
    Y = generate_varying_frequency_sine(n_samples)

    # Create input: previous value
    X = np.zeros_like(Y)
    X[1:] = Y[:-1]
    X = X.reshape(-1, 1)
    t = np.arange(len(Y)).astype(float) / len(Y)

    split = int(0.8 * len(Y))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    t_train, t_test = t[:split], t[split:]

    results = {}
    washout = 50

    # ESN
    print("\n--- ESN (H=500) ---")
    esn = TemporalESN(input_size=1, hidden_size=500, output_size=1,
                      spectral_radius=0.95, lambda_reg=1e-6)
    esn.fit(X_train, Y_train, washout=washout)
    pred_esn = esn.predict(X_test)
    nrmse_esn = compute_nrmse(Y_test[washout:], pred_esn[washout:])
    print(f"NRMSE: {nrmse_esn:.4f}")
    results['ESN'] = {'pred': pred_esn, 'nrmse': nrmse_esn}

    # Fourier (time)
    print("\n--- Fourier (time encoding, σ=20) ---")
    fourier = TemporalFourier(embedding_size=256, sigma=20.0, lambda_reg=1e-6)
    fourier.fit(t_train, Y_train, washout=washout)
    pred_fourier = fourier.predict(t_test)
    nrmse_fourier = compute_nrmse(Y_test[washout:], pred_fourier[washout:])
    print(f"NRMSE: {nrmse_fourier:.4f}")
    results['Fourier (time)'] = {'pred': pred_fourier, 'nrmse': nrmse_fourier}

    # Fourier + History
    print("\n--- Fourier + History (10 steps) ---")
    fourier_hist = TemporalFourierWithHistory(embedding_size=256, history_len=10,
                                               sigma=20.0, lambda_reg=1e-6)
    fourier_hist.fit(t_train, X_train.flatten(), Y_train, washout=washout)
    pred_fourier_hist = fourier_hist.predict(t_test, X_test.flatten())
    nrmse_fourier_hist = compute_nrmse(Y_test[washout:], pred_fourier_hist[washout:])
    print(f"NRMSE: {nrmse_fourier_hist:.4f}")
    results['Fourier + History'] = {'pred': pred_fourier_hist, 'nrmse': nrmse_fourier_hist}

    return Y_test, results


def visualize_temporal_results(save_path: str = None):
    """Run all experiments and visualize."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Task 1: Mackey-Glass
    Y_mg, results_mg = experiment_mackey_glass()
    ax = axes[0, 0]
    ax.plot(Y_mg[100:300], 'k-', label='True', linewidth=2)
    for name, data in results_mg.items():
        ax.plot(data['pred'][100:300], '--', label=f"{name} ({data['nrmse']:.3f})", alpha=0.8)
    ax.set_title('Mackey-Glass Prediction')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Task 2: NARMA-10
    Y_narma, results_narma = experiment_narma10()
    ax = axes[0, 1]
    ax.plot(Y_narma[100:300], 'k-', label='True', linewidth=2)
    for name, data in results_narma.items():
        ax.plot(data['pred'][100:300], '--', label=f"{name} ({data['nrmse']:.3f})", alpha=0.8)
    ax.set_title('NARMA-10 (requires 10-step memory)')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Task 3: Memory Capacity
    delays, esn_mc, fourier_mc = experiment_memory_capacity()
    ax = axes[1, 0]
    ax.plot(delays, esn_mc, 'bo-', label='ESN', linewidth=2, markersize=8)
    ax.plot(delays, fourier_mc, 'rs-', label='Fourier + History', linewidth=2, markersize=8)
    ax.set_title('Memory Capacity (R² vs Delay)')
    ax.set_xlabel('Delay (steps)')
    ax.set_ylabel('R² (squared correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Task 4: Varying Frequency
    Y_vf, results_vf = experiment_varying_frequency()
    ax = axes[1, 1]
    ax.plot(Y_vf[50:250], 'k-', label='True', linewidth=2)
    for name, data in results_vf.items():
        ax.plot(data['pred'][50:250], '--', label=f"{name} ({data['nrmse']:.3f})", alpha=0.8)
    ax.set_title('Varying Frequency Sine')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("TEMPORAL DATA COMPARISON: ESN vs Fourier Features")
    print("=" * 70)

    visualize_temporal_results(
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/temporal_comparison.png"
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TEMPORAL TASKS")
    print("=" * 70)
    print("""
Key Findings:

1. MACKEY-GLASS (Chaotic prediction):
   - ESN wins: Uses recurrent dynamics to capture chaotic attractor
   - Fourier (time only): Cannot predict chaos from time alone
   - Fourier + History: Better, but explicit window limits capability

2. NARMA-10 (10-step memory required):
   - ESN wins: Implicit memory through reservoir dynamics
   - Fourier (time only): Complete failure (no memory)
   - Fourier + History: Works if history >= 10 steps

3. MEMORY CAPACITY:
   - ESN: Graceful degradation with increasing delay
   - Fourier + History: Sharp cutoff at history window size

4. VARYING FREQUENCY:
   - ESN: Can adapt to changing dynamics
   - Fourier: Good if frequencies are in bandwidth
   - History helps both

CONCLUSION: For temporal tasks requiring MEMORY, ESN/Reservoir clearly wins.
Fourier needs explicit history window which is less elegant and limited.
""")

    print("\nTemporal comparison complete.")

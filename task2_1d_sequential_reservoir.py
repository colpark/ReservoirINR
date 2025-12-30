"""
Task 2: 1D Sequential Reservoir (True ESN) for INR

This IS a true reservoir with recurrent dynamics.
We process 1D coordinates sequentially, building up reservoir state.

Architecture:
    For 1D signal f(x) where x ∈ [0, 1], coordinates processed in order:

    h(0) = 0  (initial state)

    For each x_i (in sorted order):
        h(i) = (1-α)·h(i-1) + α·tanh(W_in·x_i + W_hh·h(i-1) + b)
        output(i) = W_out · h(i)

Key difference from Task 1:
    - W_hh: Recurrent weights (fixed, spectral-radius scaled)
    - Sequential processing: h(x_i) depends on h(x_{i-1})
    - Reservoir state at position x "knows" about all x' < x

Components:
    - W_in: Fixed random, shape (hidden_size, 1)
    - W_hh: Fixed random recurrent, shape (hidden_size, hidden_size), spectral-scaled
    - b: Fixed random bias, shape (hidden_size,)
    - W_out: ONLY trainable part, shape (1, hidden_size), ridge regression

Hyperparameters:
    - spectral_radius: Controls eigenvalue magnitudes of W_hh (typically 0.9)
    - leaking_rate (α): Memory retention (1.0 = no leakage)
    - density: Sparsity of W_hh (1.0 = fully connected)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class SequentialReservoirINR1D:
    """
    1D Sequential Reservoir for Implicit Neural Representation.

    True ESN with recurrent dynamics - processes coordinates sequentially.
    """

    def __init__(
        self,
        hidden_size: int,
        spectral_radius: float = 0.9,
        leaking_rate: float = 1.0,
        density: float = 1.0,
        w_in_scale: float = 1.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize the reservoir with fixed random weights.

        Args:
            hidden_size: Number of reservoir neurons
            spectral_radius: Desired spectral radius of W_hh (controls dynamics)
                - < 1.0: Stable, fading memory (typical)
                - = 1.0: Edge of chaos
                - > 1.0: Unstable, expanding dynamics
            leaking_rate: α in the update equation (0, 1]
                - 1.0: No leakage, h(t) fully replaced
                - < 1.0: Partial update, longer memory
            density: Fraction of non-zero connections in W_hh
                - 1.0: Fully connected
                - < 1.0: Sparse (e.g., 0.1 = 10% connections)
            w_in_scale: Scaling factor for input weights
            lambda_reg: Ridge regression regularization
            seed: Random seed
        """
        self.hidden_size = hidden_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.density = density
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # ============================================================
        # FIXED WEIGHTS (never trained)
        # ============================================================

        # W_in: Input-to-hidden weights
        # Shape: (hidden_size, 1) for 1D input
        # Initialization: U(-w_in_scale, w_in_scale)
        self.W_in = np.random.uniform(-w_in_scale, w_in_scale, size=(hidden_size, 1))

        # b: Bias vector
        # Shape: (hidden_size,)
        self.b = np.random.uniform(-0.1, 0.1, size=(hidden_size,))

        # W_hh: Recurrent hidden-to-hidden weights
        # Shape: (hidden_size, hidden_size)
        # This is the KEY difference from Task 1
        self.W_hh = self._init_recurrent_weights(hidden_size, spectral_radius, density)

        # ============================================================
        # TRAINABLE WEIGHTS
        # ============================================================
        self.W_out = None  # Shape: (1, hidden_size), set during training

        # Print initialization info
        self._print_init_info()

    def _init_recurrent_weights(
        self,
        hidden_size: int,
        spectral_radius: float,
        density: float
    ) -> np.ndarray:
        """
        Initialize recurrent weights with spectral radius scaling.

        Process:
        1. Create random matrix from U(-1, 1)
        2. Apply sparsity mask (if density < 1)
        3. Compute eigenvalues
        4. Scale to desired spectral radius

        Args:
            hidden_size: Dimension of W_hh
            spectral_radius: Target spectral radius
            density: Fraction of non-zero elements

        Returns:
            Scaled recurrent weight matrix
        """
        # Step 1: Random initialization U(-1, 1)
        W = np.random.uniform(-1, 1, size=(hidden_size, hidden_size))

        # Step 2: Apply sparsity mask
        if density < 1.0:
            mask = np.random.random((hidden_size, hidden_size)) < density
            W = W * mask

        # Step 3: Compute current spectral radius (max |eigenvalue|)
        eigenvalues = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigenvalues))

        # Step 4: Scale to desired spectral radius
        if current_sr > 0:
            W = W * (spectral_radius / current_sr)

        return W

    def _print_init_info(self):
        """Print initialization information."""
        # Verify actual spectral radius
        actual_sr = np.max(np.abs(np.linalg.eigvals(self.W_hh)))

        print(f"Reservoir initialized:")
        print(f"  W_in shape: {self.W_in.shape} (fixed)")
        print(f"  W_hh shape: {self.W_hh.shape} (fixed, recurrent)")
        print(f"  b shape: {self.b.shape} (fixed)")
        print(f"  Spectral radius (target): {self.spectral_radius}")
        print(f"  Spectral radius (actual): {actual_sr:.4f}")
        print(f"  Leaking rate: {self.leaking_rate}")
        print(f"  Density: {self.density}")
        print(f"  W_hh non-zero: {np.count_nonzero(self.W_hh)} / {self.W_hh.size}")
        print(f"  Total fixed params: {self.W_in.size + self.W_hh.size + self.b.size}")
        print(f"  Trainable params: {self.hidden_size} (W_out)")

    def forward_sequential(
        self,
        x_coords: np.ndarray,
        return_all_states: bool = True
    ) -> np.ndarray:
        """
        Process coordinates sequentially through reservoir.

        IMPORTANT: x_coords must be in sorted order for meaningful dynamics!

        Update equation (leaky integration):
            h(t) = (1-α)·h(t-1) + α·tanh(W_in·x(t) + W_hh·h(t-1) + b)

        Args:
            x_coords: 1D coordinates, shape (n_samples,), MUST BE SORTED
            return_all_states: If True, return all hidden states

        Returns:
            Hidden states, shape (n_samples, hidden_size)
        """
        n_samples = len(x_coords)
        α = self.leaking_rate

        # Initialize hidden state to zeros
        h = np.zeros(self.hidden_size)

        # Storage for all hidden states
        all_states = np.zeros((n_samples, self.hidden_size))

        # Process sequentially
        for t in range(n_samples):
            # Get current input (reshape to column vector)
            x_t = x_coords[t:t+1]  # Shape: (1,)

            # Compute pre-activation
            # W_in @ x_t: (hidden_size, 1) @ (1,) -> need to handle shapes
            input_contrib = (self.W_in @ x_t).flatten()  # (hidden_size,)
            recurrent_contrib = self.W_hh @ h  # (hidden_size,)

            pre_activation = input_contrib + recurrent_contrib + self.b

            # Leaky integration update
            # h_new = (1-α)·h_old + α·tanh(pre_activation)
            h = (1 - α) * h + α * np.tanh(pre_activation)

            # Store state
            all_states[t] = h

        return all_states

    def fit(self, x_coords: np.ndarray, targets: np.ndarray, washout: int = 0):
        """
        Train output weights using ridge regression.

        Args:
            x_coords: 1D coordinates, shape (n_samples,), MUST BE SORTED
            targets: Target values, shape (n_samples,) or (n_samples, 1)
            washout: Number of initial states to discard (reservoir warmup)
        """
        print(f"\nTraining...")
        print(f"  Input coords: {x_coords.shape}, range [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Washout: {washout} samples")

        # Ensure targets are 2D
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        # Step 1: Forward pass to get all hidden states
        H = self.forward_sequential(x_coords)  # (n_samples, hidden_size)
        print(f"  Hidden states shape: {H.shape}")

        # Step 2: Apply washout (discard initial transient)
        H_train = H[washout:]
        Y_train = targets[washout:]
        print(f"  After washout: H={H_train.shape}, Y={Y_train.shape}")

        # Step 3: Ridge regression
        # Solve: min_W ||H @ W^T - Y||^2 + λ||W||^2
        HTH = H_train.T @ H_train
        HTY = H_train.T @ Y_train

        reg_matrix = HTH + self.lambda_reg * np.eye(self.hidden_size)
        W_out_T = np.linalg.solve(reg_matrix, HTY)

        self.W_out = W_out_T.T  # (output_dim, hidden_size)
        print(f"  W_out shape: {self.W_out.shape}")
        print(f"  Training complete!")

    def predict(self, x_coords: np.ndarray) -> np.ndarray:
        """
        Predict values for coordinates.

        IMPORTANT: For consistent results, x_coords should be sorted.
        The reservoir state depends on the sequence order!

        Args:
            x_coords: 1D coordinates, shape (n_samples,)

        Returns:
            Predictions, shape (n_samples,) or (n_samples, output_dim)
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Get hidden states
        H = self.forward_sequential(x_coords)

        # Linear output
        output = H @ self.W_out.T

        return output.flatten() if output.shape[1] == 1 else output


def create_1d_test_signal(n_points: int = 256, signal_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 1D test signal with various frequency components.

    Args:
        n_points: Number of sample points
        signal_type: Type of signal
            - 'low': Low frequency only
            - 'high': High frequency only
            - 'mixed': Mix of low, medium, high frequencies

    Returns:
        x: Coordinates, shape (n_points,)
        y: Signal values, shape (n_points,)
    """
    x = np.linspace(0, 1, n_points)

    if signal_type == 'low':
        # Low frequency: f=2
        y = np.sin(2 * np.pi * 2 * x)

    elif signal_type == 'high':
        # High frequency: f=16
        y = np.sin(2 * np.pi * 16 * x)

    elif signal_type == 'mixed':
        # Mix of frequencies: f=2, f=8, f=16
        low = 0.5 * np.sin(2 * np.pi * 2 * x)      # f=2
        mid = 0.3 * np.sin(2 * np.pi * 8 * x)      # f=8
        high = 0.2 * np.sin(2 * np.pi * 16 * x)    # f=16
        y = low + mid + high

    elif signal_type == 'step':
        # Step function (discontinuity)
        y = np.where(x < 0.5, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown signal type: {signal_type}")

    return x, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MSE and PSNR."""
    mse = np.mean((y_true - y_pred) ** 2)

    # For PSNR, use signal range
    signal_range = y_true.max() - y_true.min()
    if signal_range == 0:
        signal_range = 1.0
    psnr = 10 * np.log10(signal_range**2 / mse) if mse > 0 else float('inf')

    return {'mse': mse, 'psnr': psnr}


def run_experiment(
    signal_type: str = 'mixed',
    n_points: int = 256,
    hidden_sizes: list = [64, 256, 512],
    spectral_radii: list = [0.9],
    leaking_rate: float = 1.0
):
    """
    Run experiment comparing different reservoir configurations.
    """
    print("=" * 70)
    print("Task 2: 1D Sequential Reservoir for INR")
    print("=" * 70)
    print(f"\nSignal type: {signal_type}")
    print(f"Points: {n_points}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Spectral radii: {spectral_radii}")
    print(f"Leaking rate: {leaking_rate}")

    # Create test signal
    x, y = create_1d_test_signal(n_points, signal_type)
    print(f"\nSignal range: [{y.min():.3f}, {y.max():.3f}]")

    results = []

    for hidden_size in hidden_sizes:
        for sr in spectral_radii:
            print(f"\n{'='*70}")
            print(f"Config: hidden_size={hidden_size}, spectral_radius={sr}")
            print(f"{'='*70}")

            # Create model
            model = SequentialReservoirINR1D(
                hidden_size=hidden_size,
                spectral_radius=sr,
                leaking_rate=leaking_rate,
                lambda_reg=1e-6
            )

            # Train (coordinates must be sorted - they already are)
            washout = min(10, n_points // 10)  # 10 samples or 10% warmup
            model.fit(x, y, washout=washout)

            # Predict
            y_pred = model.predict(x)

            # Metrics
            metrics = compute_metrics(y[washout:], y_pred[washout:])
            print(f"\nResults (excluding washout):")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  PSNR: {metrics['psnr']:.2f} dB")

            results.append({
                'hidden_size': hidden_size,
                'spectral_radius': sr,
                'y_pred': y_pred,
                'metrics': metrics
            })

    return x, y, results


def visualize_results(x, y, results, save_path: str = None):
    """Visualize signal reconstruction results."""
    n_results = len(results)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: Signal comparison
    ax1 = axes[0]
    ax1.plot(x, y, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_results))
    for i, res in enumerate(results):
        label = f"H={res['hidden_size']}, SR={res['spectral_radius']}"
        ax1.plot(x, res['y_pred'], '--', color=colors[i],
                linewidth=1.5, label=f"{label} (PSNR={res['metrics']['psnr']:.1f}dB)")

    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('1D Signal Reconstruction with Sequential Reservoir')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: Error plot
    ax2 = axes[1]
    for i, res in enumerate(results):
        error = np.abs(y - res['y_pred'])
        label = f"H={res['hidden_size']}"
        ax2.plot(x, error, color=colors[i], linewidth=1, label=label, alpha=0.7)

    ax2.set_xlabel('x')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Error')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


def compare_spectral_radii(signal_type: str = 'mixed', n_points: int = 256):
    """
    Compare different spectral radii to understand dynamics.
    """
    print("\n" + "=" * 70)
    print("Comparing Spectral Radii")
    print("=" * 70)

    x, y = create_1d_test_signal(n_points, signal_type)

    spectral_radii = [0.5, 0.9, 0.95, 0.99]
    hidden_size = 256

    results = []
    for sr in spectral_radii:
        model = SequentialReservoirINR1D(
            hidden_size=hidden_size,
            spectral_radius=sr,
            leaking_rate=1.0,
            lambda_reg=1e-6
        )

        model.fit(x, y, washout=10)
        y_pred = model.predict(x)
        metrics = compute_metrics(y[10:], y_pred[10:])

        results.append({
            'spectral_radius': sr,
            'y_pred': y_pred,
            'metrics': metrics
        })

        print(f"SR={sr}: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}")

    return x, y, results


def compare_with_task1(signal_type: str = 'mixed', n_points: int = 256, hidden_size: int = 256):
    """
    Direct comparison with Task 1 (static random features) on same 1D signal.
    """
    print("\n" + "=" * 70)
    print("Comparison: Task 1 (Static) vs Task 2 (Sequential Reservoir)")
    print("=" * 70)

    x, y = create_1d_test_signal(n_points, signal_type)

    # Task 1: Static random features (adapted for 1D)
    np.random.seed(42)
    W_in_static = np.random.randn(hidden_size, 1) / np.sqrt(1)
    b_static = np.random.uniform(-1, 1, hidden_size)

    # Forward pass (no recurrence)
    H_static = np.tanh(W_in_static @ x.reshape(1, -1) + b_static.reshape(-1, 1)).T

    # Ridge regression
    HTH = H_static.T @ H_static + 1e-6 * np.eye(hidden_size)
    HTY = H_static.T @ y.reshape(-1, 1)
    W_out_static = np.linalg.solve(HTH, HTY)
    y_pred_static = (H_static @ W_out_static).flatten()

    metrics_static = compute_metrics(y, y_pred_static)
    print(f"\nTask 1 (Static): PSNR={metrics_static['psnr']:.2f} dB")

    # Task 2: Sequential reservoir
    model = SequentialReservoirINR1D(
        hidden_size=hidden_size,
        spectral_radius=0.9,
        leaking_rate=1.0,
        lambda_reg=1e-6
    )
    model.fit(x, y, washout=10)
    y_pred_seq = model.predict(x)

    metrics_seq = compute_metrics(y[10:], y_pred_seq[10:])
    print(f"Task 2 (Sequential): PSNR={metrics_seq['psnr']:.2f} dB")

    return {
        'x': x, 'y': y,
        'static': {'pred': y_pred_static, 'metrics': metrics_static},
        'sequential': {'pred': y_pred_seq, 'metrics': metrics_seq}
    }


if __name__ == "__main__":
    # Main experiment
    x, y, results = run_experiment(
        signal_type='mixed',
        n_points=256,
        hidden_sizes=[64, 256, 512],
        spectral_radii=[0.9],
        leaking_rate=1.0
    )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Hidden Size':<15} {'Spectral Radius':<18} {'PSNR (dB)':<15} {'MSE':<15}")
    print("-" * 63)
    for res in results:
        print(f"{res['hidden_size']:<15} {res['spectral_radius']:<18} "
              f"{res['metrics']['psnr']:<15.2f} {res['metrics']['mse']:<15.6f}")

    # Visualize
    visualize_results(
        x, y, results,
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/task2_results.png"
    )

    # Compare with Task 1
    comparison = compare_with_task1(signal_type='mixed', n_points=256, hidden_size=256)

    # Visualize comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(comparison['x'], comparison['y'], 'k-', linewidth=2, label='Ground Truth')
    ax.plot(comparison['x'], comparison['static']['pred'], 'r--', linewidth=1.5,
            label=f"Task 1 Static (PSNR={comparison['static']['metrics']['psnr']:.1f}dB)")
    ax.plot(comparison['x'], comparison['sequential']['pred'], 'b--', linewidth=1.5,
            label=f"Task 2 Sequential (PSNR={comparison['sequential']['metrics']['psnr']:.1f}dB)")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Task 1 vs Task 2: Static vs Sequential Reservoir')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/task2_comparison.png", dpi=150)
    print("\nComparison figure saved to: task2_comparison.png")
    plt.show()

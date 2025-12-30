"""
Task 3: Self-Recurrent Coordinate Settling for INR

Key insight from Task 2: Sequential processing violates INR requirement that
f(x) should depend ONLY on x, not on processing history.

Solution: For each coordinate, run reservoir dynamics until they SETTLE to a
fixed point (or near-fixed point). This way:
    - Each coordinate is processed INDEPENDENTLY
    - Reservoir state is a pure function of the coordinate
    - Multiple iterations act like "depth" but with weight sharing

Architecture:
    For each coordinate [x, y]:

    h⁽⁰⁾ = 0  (or random initialization)

    For k = 1 to K:
        h⁽ᵏ⁾ = (1-α)·h⁽ᵏ⁻¹⁾ + α·tanh(W_in·[x,y] + W_hh·h⁽ᵏ⁻¹⁾ + b)
        └── same input [x,y] fed repeatedly ──┘

    output = W_out · h⁽ᴷ⁾

Key properties:
    - With spectral_radius < 1, dynamics are contractive → converge to fixed point
    - Fixed point h* satisfies: h* = (1-α)h* + α·tanh(W_in·[x,y] + W_hh·h* + b)
    - Different [x,y] → different fixed points → unique encoding
    - K iterations ≈ effective depth, but with tied weights

Relation to other methods:
    - Deep Equilibrium Models (DEQ): Find fixed point of implicit layer
    - Iterative refinement networks
    - Recurrent inference machines
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from PIL import Image


class SelfRecurrentReservoirINR:
    """
    Self-Recurrent Reservoir for INR with coordinate settling.

    For each coordinate, iterates reservoir dynamics until convergence.
    Each coordinate maps to a unique fixed point (or attractor).
    """

    def __init__(
        self,
        hidden_size: int,
        input_dim: int = 2,
        output_dim: int = 3,
        n_iterations: int = 10,
        spectral_radius: float = 0.9,
        leaking_rate: float = 1.0,
        w_in_scale: float = 1.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize the self-recurrent reservoir.

        Args:
            hidden_size: Number of reservoir neurons
            input_dim: Dimension of input coordinates (2 for 2D images)
            output_dim: Dimension of output (3 for RGB, 1 for grayscale)
            n_iterations: Number of settling iterations K
            spectral_radius: Controls contraction rate
                - < 1.0: Guaranteed convergence (contractive)
                - = 1.0: Edge of stability
                - > 1.0: May not converge
            leaking_rate: α in update equation
            w_in_scale: Input weight scaling
            lambda_reg: Ridge regression regularization
            seed: Random seed
        """
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_iterations = n_iterations
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # ============================================================
        # FIXED WEIGHTS (never trained)
        # ============================================================

        # W_in: Input-to-hidden weights
        # Shape: (hidden_size, input_dim)
        self.W_in = np.random.uniform(-w_in_scale, w_in_scale,
                                       size=(hidden_size, input_dim))

        # b: Bias vector
        # Shape: (hidden_size,)
        self.b = np.random.uniform(-0.1, 0.1, size=(hidden_size,))

        # W_hh: Recurrent weights (spectral-radius scaled)
        # Shape: (hidden_size, hidden_size)
        self.W_hh = self._init_recurrent_weights(hidden_size, spectral_radius)

        # ============================================================
        # TRAINABLE WEIGHTS
        # ============================================================
        self.W_out = None  # Shape: (output_dim, hidden_size)

        self._print_init_info()

    def _init_recurrent_weights(self, hidden_size: int, spectral_radius: float) -> np.ndarray:
        """Initialize recurrent weights with spectral radius scaling."""
        W = np.random.uniform(-1, 1, size=(hidden_size, hidden_size))

        # Compute current spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigenvalues))

        # Scale to desired spectral radius
        if current_sr > 0:
            W = W * (spectral_radius / current_sr)

        return W

    def _print_init_info(self):
        """Print initialization summary."""
        actual_sr = np.max(np.abs(np.linalg.eigvals(self.W_hh)))

        print(f"Self-Recurrent Reservoir initialized:")
        print(f"  W_in shape: {self.W_in.shape} (fixed)")
        print(f"  W_hh shape: {self.W_hh.shape} (fixed, recurrent)")
        print(f"  b shape: {self.b.shape} (fixed)")
        print(f"  Spectral radius (target): {self.spectral_radius}")
        print(f"  Spectral radius (actual): {actual_sr:.4f}")
        print(f"  Leaking rate: {self.leaking_rate}")
        print(f"  Settling iterations: {self.n_iterations}")
        print(f"  Total fixed params: {self.W_in.size + self.W_hh.size + self.b.size}")
        print(f"  Trainable params: {self.output_dim * self.hidden_size} (W_out)")

    def compute_hidden_single(self, coord: np.ndarray, return_trajectory: bool = False):
        """
        Compute hidden state for a single coordinate via iterative settling.

        Args:
            coord: Single coordinate, shape (input_dim,)
            return_trajectory: If True, return all intermediate states

        Returns:
            Final hidden state (hidden_size,) or trajectory (n_iterations+1, hidden_size)
        """
        α = self.leaking_rate

        # Initialize hidden state
        h = np.zeros(self.hidden_size)

        # Pre-compute input contribution (constant across iterations)
        # W_in @ coord: (hidden_size,)
        input_contrib = self.W_in @ coord  # (hidden_size,)

        if return_trajectory:
            trajectory = [h.copy()]

        # Iterate K times with SAME input
        for k in range(self.n_iterations):
            # Recurrent contribution: W_hh @ h
            recurrent_contrib = self.W_hh @ h  # (hidden_size,)

            # Pre-activation: input + recurrent + bias
            pre_activation = input_contrib + recurrent_contrib + self.b

            # Leaky integration update
            h = (1 - α) * h + α * np.tanh(pre_activation)

            if return_trajectory:
                trajectory.append(h.copy())

        if return_trajectory:
            return np.array(trajectory)
        return h

    def compute_hidden_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute hidden states for batch of coordinates.

        Each coordinate is processed INDEPENDENTLY through K iterations.

        Args:
            coords: Coordinates, shape (n_samples, input_dim)

        Returns:
            Hidden states, shape (n_samples, hidden_size)
        """
        n_samples = coords.shape[0]
        α = self.leaking_rate

        # Initialize all hidden states to zero
        H = np.zeros((n_samples, self.hidden_size))

        # Pre-compute input contributions for all coordinates
        # coords: (n_samples, input_dim)
        # W_in: (hidden_size, input_dim)
        # input_contrib: (n_samples, hidden_size)
        input_contrib = coords @ self.W_in.T  # (n_samples, hidden_size)

        # Iterate K times
        for k in range(self.n_iterations):
            # Recurrent contribution for all samples
            # H: (n_samples, hidden_size)
            # W_hh: (hidden_size, hidden_size)
            # recurrent_contrib: (n_samples, hidden_size)
            recurrent_contrib = H @ self.W_hh.T

            # Pre-activation: input + recurrent + bias
            pre_activation = input_contrib + recurrent_contrib + self.b  # Broadcasting bias

            # Leaky integration update (vectorized)
            H = (1 - α) * H + α * np.tanh(pre_activation)

        return H

    def fit(self, coords: np.ndarray, targets: np.ndarray):
        """
        Train output weights via ridge regression.

        Args:
            coords: Coordinates, shape (n_samples, input_dim)
            targets: Target values, shape (n_samples, output_dim)
        """
        print(f"\nTraining...")
        print(f"  Coords shape: {coords.shape}")
        print(f"  Targets shape: {targets.shape}")

        # Get hidden states (after settling)
        H = self.compute_hidden_batch(coords)
        print(f"  Hidden states shape: {H.shape}")

        # Ridge regression
        HTH = H.T @ H
        HTY = H.T @ targets

        reg_matrix = HTH + self.lambda_reg * np.eye(self.hidden_size)
        W_out_T = np.linalg.solve(reg_matrix, HTY)

        self.W_out = W_out_T.T
        print(f"  W_out shape: {self.W_out.shape}")
        print(f"  Training complete!")

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict output for coordinates."""
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        H = self.compute_hidden_batch(coords)
        return H @ self.W_out.T

    def predict_clipped(self, coords: np.ndarray) -> np.ndarray:
        """Predict and clip to valid range [0, 1]."""
        return np.clip(self.predict(coords), 0, 1)


def analyze_convergence(model: SelfRecurrentReservoirINR, n_test_coords: int = 5):
    """
    Analyze convergence behavior for random test coordinates.

    Plots the trajectory of hidden state norm across iterations.
    """
    print("\n" + "=" * 60)
    print("Convergence Analysis")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Generate random test coordinates
    np.random.seed(123)
    test_coords = np.random.uniform(-1, 1, size=(n_test_coords, model.input_dim))

    # Track state norms and changes
    for i, coord in enumerate(test_coords):
        trajectory = model.compute_hidden_single(coord, return_trajectory=True)

        # State norms
        norms = np.linalg.norm(trajectory, axis=1)
        axes[0].plot(norms, label=f'Coord {i+1}', marker='o', markersize=3)

        # State changes (convergence metric)
        changes = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        axes[1].plot(changes, label=f'Coord {i+1}', marker='o', markersize=3)

        print(f"Coord {i+1}: ||h_final||={norms[-1]:.4f}, Δh_final={changes[-1]:.6f}")

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('||h||')
    axes[0].set_title('Hidden State Norm')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('||h^(k) - h^(k-1)||')
    axes[1].set_title('Convergence (State Change)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    plt.tight_layout()
    return fig


def create_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Create synthetic test image with various frequencies."""
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Mix of frequencies
    low_freq = (xx + yy) / 2
    med_freq = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * xx) * np.sin(2 * np.pi * 4 * yy)

    img = np.stack([low_freq, med_freq, 0.5 * np.ones_like(xx)], axis=-1)
    return img


def create_coordinate_grid(height: int, width: int) -> np.ndarray:
    """Create normalized coordinate grid [-1, 1] x [-1, 1]."""
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    return coords


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def run_experiment(
    image: np.ndarray,
    hidden_sizes: list = [256, 512],
    n_iterations_list: list = [5, 10, 20],
    spectral_radius: float = 0.9
):
    """
    Run experiment varying hidden size and number of iterations.
    """
    print("=" * 70)
    print("Task 3: Self-Recurrent Coordinate Settling for INR")
    print("=" * 70)

    height, width, _ = image.shape
    n_pixels = height * width

    print(f"\nImage size: {height} x {width} = {n_pixels} pixels")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Iterations: {n_iterations_list}")
    print(f"Spectral radius: {spectral_radius}")

    # Prepare data
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    results = []

    for hidden_size in hidden_sizes:
        for n_iter in n_iterations_list:
            print(f"\n{'='*70}")
            print(f"Config: hidden_size={hidden_size}, iterations={n_iter}")
            print(f"{'='*70}")

            model = SelfRecurrentReservoirINR(
                hidden_size=hidden_size,
                input_dim=2,
                output_dim=3,
                n_iterations=n_iter,
                spectral_radius=spectral_radius,
                leaking_rate=1.0,
                lambda_reg=1e-6
            )

            # Train
            model.fit(coords, targets)

            # Predict
            predictions = model.predict_clipped(coords)
            pred_image = predictions.reshape(height, width, 3)

            # Metrics
            psnr = compute_psnr(image, pred_image)
            mse = np.mean((image - pred_image) ** 2)

            print(f"\nResults:")
            print(f"  MSE: {mse:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")

            results.append({
                'hidden_size': hidden_size,
                'n_iterations': n_iter,
                'pred_image': pred_image,
                'psnr': psnr,
                'mse': mse,
                'model': model
            })

    return results


def compare_with_task1(image: np.ndarray, hidden_size: int = 256, n_iterations: int = 10):
    """
    Direct comparison with Task 1 (static random features).
    """
    print("\n" + "=" * 70)
    print("Comparison: Task 1 (Static) vs Task 3 (Self-Recurrent)")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    # Task 1: Static (no recurrence, K=1 iteration essentially)
    print("\nTask 1: Static Random Features")
    model_static = SelfRecurrentReservoirINR(
        hidden_size=hidden_size,
        input_dim=2,
        output_dim=3,
        n_iterations=1,  # Single pass, no recurrence effect
        spectral_radius=0.9,
        lambda_reg=1e-6
    )
    model_static.fit(coords, targets)
    pred_static = model_static.predict_clipped(coords).reshape(height, width, 3)
    psnr_static = compute_psnr(image, pred_static)
    print(f"PSNR: {psnr_static:.2f} dB")

    # Task 3: Self-recurrent settling
    print(f"\nTask 3: Self-Recurrent ({n_iterations} iterations)")
    model_settling = SelfRecurrentReservoirINR(
        hidden_size=hidden_size,
        input_dim=2,
        output_dim=3,
        n_iterations=n_iterations,
        spectral_radius=0.9,
        lambda_reg=1e-6
    )
    model_settling.fit(coords, targets)
    pred_settling = model_settling.predict_clipped(coords).reshape(height, width, 3)
    psnr_settling = compute_psnr(image, pred_settling)
    print(f"PSNR: {psnr_settling:.2f} dB")

    return {
        'static': {'pred': pred_static, 'psnr': psnr_static},
        'settling': {'pred': pred_settling, 'psnr': psnr_settling}
    }


def visualize_results(image: np.ndarray, results: list, save_path: str = None):
    """Visualize reconstruction results."""
    # Group by hidden size
    hidden_sizes = sorted(set(r['hidden_size'] for r in results))
    n_iters = sorted(set(r['n_iterations'] for r in results))

    n_rows = len(hidden_sizes)
    n_cols = len(n_iters) + 1  # +1 for original

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, hs in enumerate(hidden_sizes):
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].set_ylabel(f'H={hs}')
        axes[i, 0].axis('off')

        # Results for this hidden size
        for j, n_iter in enumerate(n_iters):
            res = next(r for r in results if r['hidden_size'] == hs and r['n_iterations'] == n_iter)
            axes[i, j + 1].imshow(res['pred_image'])
            axes[i, j + 1].set_title(f'K={n_iter}\n{res["psnr"]:.1f}dB' if i == 0 else f'{res["psnr"]:.1f}dB')
            axes[i, j + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def study_iteration_effect(image: np.ndarray, hidden_size: int = 256, max_iter: int = 50):
    """
    Study how PSNR changes with number of iterations.
    """
    print("\n" + "=" * 70)
    print(f"Iteration Study: hidden_size={hidden_size}")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    iterations = [1, 2, 5, 10, 15, 20, 30, 40, 50]
    psnrs = []

    for n_iter in iterations:
        model = SelfRecurrentReservoirINR(
            hidden_size=hidden_size,
            input_dim=2,
            output_dim=3,
            n_iterations=n_iter,
            spectral_radius=0.9,
            lambda_reg=1e-6
        )
        model.fit(coords, targets)
        pred = model.predict_clipped(coords).reshape(height, width, 3)
        psnr = compute_psnr(image, pred)
        psnrs.append(psnr)
        print(f"K={n_iter:3d}: PSNR={psnr:.2f} dB")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, psnrs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Iterations (K)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Effect of Settling Iterations (H={hidden_size})')
    ax.grid(True, alpha=0.3)

    return iterations, psnrs, fig


if __name__ == "__main__":
    # Create test image
    print("Creating test image...")
    image = create_test_image(height=64, width=64)
    print(f"Image shape: {image.shape}")

    # Main experiment
    results = run_experiment(
        image=image,
        hidden_sizes=[256, 512],
        n_iterations_list=[1, 5, 10, 20],
        spectral_radius=0.9
    )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Hidden Size':<15} {'Iterations':<12} {'PSNR (dB)':<15} {'MSE':<15}")
    print("-" * 57)
    for res in results:
        print(f"{res['hidden_size']:<15} {res['n_iterations']:<12} "
              f"{res['psnr']:<15.2f} {res['mse']:<15.6f}")

    # Visualize
    visualize_results(
        image, results,
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/task3_results.png"
    )

    # Convergence analysis for best model
    best_model = max(results, key=lambda x: x['psnr'])['model']
    conv_fig = analyze_convergence(best_model)
    conv_fig.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/task3_convergence.png",
                     dpi=150, bbox_inches='tight')
    print("\nConvergence figure saved.")

    # Iteration study
    iters, psnrs, iter_fig = study_iteration_effect(image, hidden_size=256, max_iter=50)
    iter_fig.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/task3_iteration_study.png",
                     dpi=150, bbox_inches='tight')
    print("Iteration study figure saved.")

    # Comparison with Task 1
    comparison = compare_with_task1(image, hidden_size=256, n_iterations=20)
    print(f"\nFinal Comparison:")
    print(f"  Task 1 (K=1): {comparison['static']['psnr']:.2f} dB")
    print(f"  Task 3 (K=20): {comparison['settling']['psnr']:.2f} dB")
    print(f"  Improvement: {comparison['settling']['psnr'] - comparison['static']['psnr']:.2f} dB")

    print("\nAll figures saved. Task 3 complete.")

"""
Task 5: Multi-Reservoir Ensemble for INR

Key insight from Task 3: Self-recurrent settling creates rich basis functions,
but a single spectral radius may limit the "frequency range" of representable functions.

Solution: Use MULTIPLE reservoirs with DIFFERENT spectral radii.
Each reservoir captures different dynamical characteristics:
    - Low spectral radius (0.5): Fast settling, local/high-frequency features
    - Medium spectral radius (0.9): Balanced dynamics
    - High spectral radius (0.99): Slow settling, global/low-frequency features

This is analogous to:
    - Multi-scale Fourier features (different σ values)
    - Wavelet decomposition (different scales)
    - Deep networks with skip connections (different receptive fields)

Architecture:
    For each coordinate [x, y]:

    Reservoir 1 (SR=0.5):  h₁ = settle(coord, W_hh₁)  # Fast dynamics
    Reservoir 2 (SR=0.9):  h₂ = settle(coord, W_hh₂)  # Medium dynamics
    Reservoir 3 (SR=0.99): h₃ = settle(coord, W_hh₃)  # Slow dynamics

    h_combined = concat(h₁, h₂, h₃)  # Shape: (3 * hidden_size,)

    output = W_out · h_combined  # Single linear readout

Key properties:
    - Each reservoir has INDEPENDENT random weights
    - Different spectral radii create DIVERSE basis functions
    - Concatenation preserves all information
    - Single ridge regression trains the readout
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class SingleReservoir:
    """
    A single reservoir module for the ensemble.
    Implements self-recurrent settling (from Task 3).
    """

    def __init__(
        self,
        hidden_size: int,
        input_dim: int,
        spectral_radius: float,
        n_iterations: int,
        leaking_rate: float = 1.0,
        w_in_scale: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Initialize a single reservoir.

        Args:
            hidden_size: Number of neurons in this reservoir
            input_dim: Dimension of input coordinates
            spectral_radius: Target spectral radius for W_hh
            n_iterations: Number of settling iterations
            leaking_rate: Leaking rate α
            w_in_scale: Input weight scaling
            seed: Random seed (None for random)
        """
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.n_iterations = n_iterations
        self.leaking_rate = leaking_rate

        if seed is not None:
            np.random.seed(seed)

        # Initialize weights
        self.W_in = np.random.uniform(-w_in_scale, w_in_scale,
                                       size=(hidden_size, input_dim))
        self.b = np.random.uniform(-0.1, 0.1, size=(hidden_size,))
        self.W_hh = self._init_recurrent_weights(hidden_size, spectral_radius)

        # Verify spectral radius
        self.actual_sr = np.max(np.abs(np.linalg.eigvals(self.W_hh)))

    def _init_recurrent_weights(self, hidden_size: int, spectral_radius: float) -> np.ndarray:
        """Initialize W_hh with target spectral radius."""
        W = np.random.uniform(-1, 1, size=(hidden_size, hidden_size))
        eigenvalues = np.linalg.eigvals(W)
        current_sr = np.max(np.abs(eigenvalues))
        if current_sr > 0:
            W = W * (spectral_radius / current_sr)
        return W

    def compute_hidden_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute settled hidden states for batch of coordinates.

        Args:
            coords: Shape (n_samples, input_dim)

        Returns:
            Hidden states, shape (n_samples, hidden_size)
        """
        n_samples = coords.shape[0]
        α = self.leaking_rate

        H = np.zeros((n_samples, self.hidden_size))
        input_contrib = coords @ self.W_in.T

        for k in range(self.n_iterations):
            recurrent_contrib = H @ self.W_hh.T
            pre_activation = input_contrib + recurrent_contrib + self.b
            H = (1 - α) * H + α * np.tanh(pre_activation)

        return H


class MultiReservoirEnsemble:
    """
    Ensemble of multiple reservoirs with different spectral radii.

    Each reservoir provides a different "view" of the coordinate space,
    analogous to multi-scale analysis.
    """

    def __init__(
        self,
        hidden_size_per_reservoir: int,
        input_dim: int = 2,
        output_dim: int = 3,
        spectral_radii: List[float] = [0.5, 0.9, 0.99],
        n_iterations: int = 10,
        leaking_rate: float = 1.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize the multi-reservoir ensemble.

        Args:
            hidden_size_per_reservoir: Neurons per reservoir
            input_dim: Dimension of input coordinates
            output_dim: Dimension of output (3 for RGB)
            spectral_radii: List of spectral radii for each reservoir
            n_iterations: Settling iterations per reservoir
            leaking_rate: Leaking rate for all reservoirs
            lambda_reg: Ridge regression regularization
            seed: Base random seed
        """
        self.hidden_size_per_reservoir = hidden_size_per_reservoir
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spectral_radii = spectral_radii
        self.n_reservoirs = len(spectral_radii)
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg

        # Total hidden dimension = sum of all reservoir sizes
        self.total_hidden_size = hidden_size_per_reservoir * self.n_reservoirs

        # Create reservoirs with different seeds
        self.reservoirs = []
        for i, sr in enumerate(spectral_radii):
            reservoir = SingleReservoir(
                hidden_size=hidden_size_per_reservoir,
                input_dim=input_dim,
                spectral_radius=sr,
                n_iterations=n_iterations,
                leaking_rate=leaking_rate,
                seed=seed + i * 1000  # Different seed for each reservoir
            )
            self.reservoirs.append(reservoir)

        # Trainable output weights
        self.W_out = None

        self._print_init_info()

    def _print_init_info(self):
        """Print initialization summary."""
        print(f"Multi-Reservoir Ensemble initialized:")
        print(f"  Number of reservoirs: {self.n_reservoirs}")
        print(f"  Hidden size per reservoir: {self.hidden_size_per_reservoir}")
        print(f"  Total hidden size: {self.total_hidden_size}")
        print(f"  Spectral radii: {self.spectral_radii}")
        print(f"  Settling iterations: {self.n_iterations}")
        print(f"\n  Individual reservoirs:")
        for i, (sr, res) in enumerate(zip(self.spectral_radii, self.reservoirs)):
            print(f"    Reservoir {i+1}: SR_target={sr:.2f}, SR_actual={res.actual_sr:.4f}")

        # Count parameters
        fixed_per_res = (self.hidden_size_per_reservoir * self.input_dim +
                         self.hidden_size_per_reservoir +
                         self.hidden_size_per_reservoir ** 2)
        total_fixed = fixed_per_res * self.n_reservoirs
        trainable = self.output_dim * self.total_hidden_size

        print(f"\n  Total fixed params: {total_fixed}")
        print(f"  Trainable params: {trainable} (W_out)")

    def compute_hidden_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute concatenated hidden states from all reservoirs.

        Args:
            coords: Shape (n_samples, input_dim)

        Returns:
            Concatenated hidden states, shape (n_samples, total_hidden_size)
        """
        # Get hidden states from each reservoir
        hidden_states = []
        for reservoir in self.reservoirs:
            h = reservoir.compute_hidden_batch(coords)
            hidden_states.append(h)

        # Concatenate along hidden dimension
        H_combined = np.concatenate(hidden_states, axis=1)
        return H_combined

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

        # Get combined hidden states
        H = self.compute_hidden_batch(coords)
        print(f"  Combined hidden states shape: {H.shape}")

        # Ridge regression
        HTH = H.T @ H
        HTY = H.T @ targets
        reg_matrix = HTH + self.lambda_reg * np.eye(self.total_hidden_size)
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
        """Predict and clip to [0, 1]."""
        return np.clip(self.predict(coords), 0, 1)

    def get_per_reservoir_contribution(self, coords: np.ndarray) -> List[np.ndarray]:
        """
        Analyze contribution of each reservoir to the output.

        Returns predictions if only that reservoir's weights were used.
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained.")

        contributions = []
        start_idx = 0

        for i, reservoir in enumerate(self.reservoirs):
            h = reservoir.compute_hidden_batch(coords)
            end_idx = start_idx + self.hidden_size_per_reservoir

            # Extract weights for this reservoir
            W_out_slice = self.W_out[:, start_idx:end_idx]

            # Compute contribution
            contrib = h @ W_out_slice.T
            contributions.append(contrib)

            start_idx = end_idx

        return contributions


def create_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Create synthetic test image with various frequencies."""
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

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
    """Compute PSNR."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def run_experiment(
    image: np.ndarray,
    spectral_radii_configs: List[List[float]],
    hidden_sizes: List[int] = [128, 256],
    n_iterations: int = 10
):
    """
    Run experiment comparing different ensemble configurations.
    """
    print("=" * 70)
    print("Task 5: Multi-Reservoir Ensemble for INR")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    print(f"\nImage size: {height} x {width}")
    print(f"Testing spectral radii configs: {spectral_radii_configs}")
    print(f"Hidden sizes per reservoir: {hidden_sizes}")

    results = []

    for hidden_size in hidden_sizes:
        for sr_config in spectral_radii_configs:
            print(f"\n{'='*70}")
            print(f"Config: H={hidden_size}/reservoir, SR={sr_config}")
            print(f"{'='*70}")

            model = MultiReservoirEnsemble(
                hidden_size_per_reservoir=hidden_size,
                input_dim=2,
                output_dim=3,
                spectral_radii=sr_config,
                n_iterations=n_iterations,
                lambda_reg=1e-6
            )

            model.fit(coords, targets)

            predictions = model.predict_clipped(coords)
            pred_image = predictions.reshape(height, width, 3)

            psnr = compute_psnr(image, pred_image)
            mse = np.mean((image - pred_image) ** 2)

            print(f"\nResults:")
            print(f"  MSE: {mse:.6f}")
            print(f"  PSNR: {psnr:.2f} dB")

            results.append({
                'hidden_size': hidden_size,
                'spectral_radii': sr_config,
                'n_reservoirs': len(sr_config),
                'total_hidden': hidden_size * len(sr_config),
                'pred_image': pred_image,
                'psnr': psnr,
                'mse': mse,
                'model': model
            })

    return results


def compare_with_single_reservoir(image: np.ndarray, total_hidden: int = 512, n_iterations: int = 10):
    """
    Compare ensemble vs single reservoir with same total hidden size.
    """
    print("\n" + "=" * 70)
    print("Comparison: Single Reservoir vs Multi-Reservoir Ensemble")
    print(f"Total hidden size: {total_hidden}")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    results = {}

    # Single reservoir (Task 3 style)
    print("\n--- Single Reservoir (SR=0.9) ---")
    single_model = MultiReservoirEnsemble(
        hidden_size_per_reservoir=total_hidden,
        spectral_radii=[0.9],  # Single reservoir
        n_iterations=n_iterations,
        lambda_reg=1e-6
    )
    single_model.fit(coords, targets)
    pred_single = single_model.predict_clipped(coords).reshape(height, width, 3)
    psnr_single = compute_psnr(image, pred_single)
    print(f"PSNR: {psnr_single:.2f} dB")
    results['single'] = {'pred': pred_single, 'psnr': psnr_single}

    # Two reservoirs (256 each)
    print("\n--- Two Reservoirs (SR=0.5, 0.99) ---")
    two_model = MultiReservoirEnsemble(
        hidden_size_per_reservoir=total_hidden // 2,
        spectral_radii=[0.5, 0.99],
        n_iterations=n_iterations,
        lambda_reg=1e-6
    )
    two_model.fit(coords, targets)
    pred_two = two_model.predict_clipped(coords).reshape(height, width, 3)
    psnr_two = compute_psnr(image, pred_two)
    print(f"PSNR: {psnr_two:.2f} dB")
    results['two'] = {'pred': pred_two, 'psnr': psnr_two}

    # Three reservoirs (170 each ≈ 512)
    print("\n--- Three Reservoirs (SR=0.5, 0.9, 0.99) ---")
    three_model = MultiReservoirEnsemble(
        hidden_size_per_reservoir=total_hidden // 3,
        spectral_radii=[0.5, 0.9, 0.99],
        n_iterations=n_iterations,
        lambda_reg=1e-6
    )
    three_model.fit(coords, targets)
    pred_three = three_model.predict_clipped(coords).reshape(height, width, 3)
    psnr_three = compute_psnr(image, pred_three)
    print(f"PSNR: {psnr_three:.2f} dB")
    results['three'] = {'pred': pred_three, 'psnr': psnr_three}

    # Five reservoirs (102 each ≈ 512)
    print("\n--- Five Reservoirs (SR=0.3, 0.5, 0.7, 0.9, 0.99) ---")
    five_model = MultiReservoirEnsemble(
        hidden_size_per_reservoir=total_hidden // 5,
        spectral_radii=[0.3, 0.5, 0.7, 0.9, 0.99],
        n_iterations=n_iterations,
        lambda_reg=1e-6
    )
    five_model.fit(coords, targets)
    pred_five = five_model.predict_clipped(coords).reshape(height, width, 3)
    psnr_five = compute_psnr(image, pred_five)
    print(f"PSNR: {psnr_five:.2f} dB")
    results['five'] = {'pred': pred_five, 'psnr': psnr_five}

    return results


def analyze_reservoir_contributions(model: MultiReservoirEnsemble, coords: np.ndarray,
                                     image_shape: Tuple[int, int, int]):
    """
    Visualize what each reservoir contributes to the final output.
    """
    height, width, _ = image_shape
    contributions = model.get_per_reservoir_contribution(coords)

    n_res = len(contributions)
    fig, axes = plt.subplots(1, n_res + 1, figsize=(4 * (n_res + 1), 4))

    # Full prediction
    full_pred = model.predict_clipped(coords).reshape(height, width, 3)
    axes[0].imshow(full_pred)
    axes[0].set_title('Full Prediction')
    axes[0].axis('off')

    # Per-reservoir contributions
    for i, (contrib, sr) in enumerate(zip(contributions, model.spectral_radii)):
        # Normalize contribution for visualization
        contrib_img = contrib.reshape(height, width, 3)
        contrib_normalized = (contrib_img - contrib_img.min()) / (contrib_img.max() - contrib_img.min() + 1e-8)

        axes[i + 1].imshow(contrib_normalized)
        axes[i + 1].set_title(f'SR={sr}\ncontribution')
        axes[i + 1].axis('off')

    plt.tight_layout()
    return fig


def visualize_results(image: np.ndarray, results: list, save_path: str = None):
    """Visualize reconstruction results."""
    n_results = len(results)
    n_cols = min(4, n_results + 1)
    n_rows = (n_results + n_cols) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Results
    for i, res in enumerate(results):
        ax = axes[i + 1]
        ax.imshow(res['pred_image'])
        sr_str = ','.join([f'{sr:.1f}' for sr in res['spectral_radii']])
        ax.set_title(f'SR=[{sr_str}]\nH={res["total_hidden"]}\n{res["psnr"]:.1f}dB')
        ax.axis('off')

    # Hide unused axes
    for i in range(len(results) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def study_spectral_radius_combinations(image: np.ndarray, hidden_per_res: int = 128, n_iterations: int = 10):
    """
    Study different combinations of spectral radii.
    """
    print("\n" + "=" * 70)
    print("Spectral Radius Combination Study")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    # Different SR combinations to test
    sr_combinations = [
        [0.9],                          # Single (baseline)
        [0.5, 0.99],                    # Extreme pair
        [0.7, 0.95],                    # Moderate pair
        [0.5, 0.9, 0.99],               # Wide spread
        [0.8, 0.9, 0.95],               # Narrow spread (high)
        [0.5, 0.7, 0.9],                # Narrow spread (low-mid)
        [0.3, 0.6, 0.9, 0.99],          # Four reservoirs
        [0.5, 0.7, 0.85, 0.95, 0.99],   # Five reservoirs
    ]

    results = []
    for sr_config in sr_combinations:
        model = MultiReservoirEnsemble(
            hidden_size_per_reservoir=hidden_per_res,
            spectral_radii=sr_config,
            n_iterations=n_iterations,
            lambda_reg=1e-6
        )
        model.fit(coords, targets)
        pred = model.predict_clipped(coords).reshape(height, width, 3)
        psnr = compute_psnr(image, pred)

        total_h = hidden_per_res * len(sr_config)
        sr_str = ','.join([f'{sr:.2f}' for sr in sr_config])
        print(f"SR=[{sr_str}], Total H={total_h}: PSNR={psnr:.2f} dB")

        results.append({
            'spectral_radii': sr_config,
            'total_hidden': total_h,
            'psnr': psnr,
            'pred_image': pred
        })

    return results


if __name__ == "__main__":
    # Create test image
    print("Creating test image...")
    image = create_test_image(height=64, width=64)
    print(f"Image shape: {image.shape}")

    # Main experiment
    results = run_experiment(
        image=image,
        spectral_radii_configs=[
            [0.9],                    # Single (baseline)
            [0.5, 0.99],              # Two reservoirs
            [0.5, 0.9, 0.99],         # Three reservoirs
        ],
        hidden_sizes=[128, 256],
        n_iterations=10
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Spectral Radii':<25} {'H/res':<10} {'Total H':<10} {'PSNR (dB)':<12} {'MSE':<12}")
    print("-" * 69)
    for res in results:
        sr_str = ','.join([f'{sr:.1f}' for sr in res['spectral_radii']])
        print(f"[{sr_str}]".ljust(25) +
              f"{res['hidden_size']:<10} {res['total_hidden']:<10} "
              f"{res['psnr']:<12.2f} {res['mse']:<12.6f}")

    # Visualize
    visualize_results(
        image, results,
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/task5_results.png"
    )

    # Fair comparison (same total hidden size)
    comparison = compare_with_single_reservoir(image, total_hidden=512, n_iterations=10)

    print("\n" + "=" * 70)
    print("FAIR COMPARISON (Total H=512)")
    print("=" * 70)
    for name, data in comparison.items():
        print(f"  {name}: PSNR={data['psnr']:.2f} dB")

    # Visualize comparison
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, (name, data) in enumerate(comparison.items()):
        axes[i + 1].imshow(data['pred'])
        axes[i + 1].set_title(f'{name}\n{data["psnr"]:.1f}dB')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/task5_comparison.png",
                dpi=150, bbox_inches='tight')
    print("\nComparison figure saved.")
    plt.close()

    # Study SR combinations
    sr_study = study_spectral_radius_combinations(image, hidden_per_res=128, n_iterations=10)

    # Analyze contributions of best model
    best_result = max(results, key=lambda x: x['psnr'])
    if best_result['n_reservoirs'] > 1:
        coords = create_coordinate_grid(64, 64)
        contrib_fig = analyze_reservoir_contributions(
            best_result['model'], coords, image.shape
        )
        contrib_fig.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/task5_contributions.png",
                           dpi=150, bbox_inches='tight')
        print("Contribution analysis figure saved.")
        plt.close()

    print("\nTask 5 complete.")

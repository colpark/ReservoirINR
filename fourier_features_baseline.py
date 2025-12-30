"""
Fourier Feature Networks Baseline for Comparison

Implements the original Fourier Feature approach from:
"Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
(Tancik et al., NeurIPS 2020)

Architecture:
    Input: [x, y] coordinate (2D)
           ↓
    Fourier Encoding: γ(v) = [sin(2πBv), cos(2πBv)]
           ↓           where B ~ N(0, σ²), shape (m, 2)
           ↓           output shape: (2m,)
    MLP or Linear Readout
           ↓
    RGB output (3D)

Two variants implemented:
1. Fourier + Ridge Regression (fair comparison with reservoir methods)
2. Fourier + MLP with gradient descent (original paper approach)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class FourierFeaturesRidge:
    """
    Fourier Features with Ridge Regression readout.

    Fair comparison with reservoir methods (same training: ridge regression).
    """

    def __init__(
        self,
        embedding_size: int,
        input_dim: int = 2,
        output_dim: int = 3,
        sigma: float = 10.0,
        lambda_reg: float = 1e-6,
        seed: int = 42
    ):
        """
        Initialize Fourier feature encoder.

        Args:
            embedding_size: Number of Fourier frequencies (m)
                           Total encoding dimension = 2m (sin + cos)
            input_dim: Dimension of input coordinates
            output_dim: Dimension of output (3 for RGB)
            sigma: Standard deviation for Gaussian frequency sampling
                   Controls bandwidth of representable frequencies
                   - Small σ (1-5): Low frequencies
                   - Medium σ (10-15): Natural images
                   - Large σ (20+): High frequencies
            lambda_reg: Ridge regression regularization
            seed: Random seed
        """
        self.embedding_size = embedding_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.lambda_reg = lambda_reg

        np.random.seed(seed)

        # B: Random frequency matrix
        # Shape: (embedding_size, input_dim)
        # Each row is a random frequency vector
        # Sampled from N(0, σ²)
        self.B = np.random.randn(embedding_size, input_dim) * sigma

        # Total encoding dimension
        self.encoding_dim = 2 * embedding_size  # sin + cos

        # Trainable output weights
        self.W_out = None

        self._print_init_info()

    def _print_init_info(self):
        """Print initialization summary."""
        print(f"Fourier Features (Ridge) initialized:")
        print(f"  Embedding size (m): {self.embedding_size}")
        print(f"  Encoding dimension: {self.encoding_dim}")
        print(f"  Frequency scale (σ): {self.sigma}")
        print(f"  B matrix shape: {self.B.shape}")
        print(f"  Fixed params: {self.B.size}")
        print(f"  Trainable params: {self.output_dim * self.encoding_dim}")

    def encode(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply Fourier feature encoding.

        γ(v) = [sin(2πBv), cos(2πBv)]

        Args:
            coords: Input coordinates, shape (n_samples, input_dim)

        Returns:
            Encoded features, shape (n_samples, 2 * embedding_size)
        """
        # coords: (n_samples, input_dim)
        # B: (embedding_size, input_dim)
        # projection: (n_samples, embedding_size)
        projection = 2 * np.pi * coords @ self.B.T

        # Concatenate sin and cos
        # Shape: (n_samples, 2 * embedding_size)
        encoded = np.concatenate([np.sin(projection), np.cos(projection)], axis=1)

        return encoded

    def fit(self, coords: np.ndarray, targets: np.ndarray):
        """Train via ridge regression."""
        print(f"\nTraining...")
        print(f"  Coords shape: {coords.shape}")
        print(f"  Targets shape: {targets.shape}")

        # Encode coordinates
        H = self.encode(coords)
        print(f"  Encoded features shape: {H.shape}")

        # Ridge regression
        HTH = H.T @ H
        HTY = H.T @ targets
        reg_matrix = HTH + self.lambda_reg * np.eye(self.encoding_dim)
        W_out_T = np.linalg.solve(reg_matrix, HTY)

        self.W_out = W_out_T.T
        print(f"  W_out shape: {self.W_out.shape}")
        print(f"  Training complete!")

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict output for coordinates."""
        if self.W_out is None:
            raise RuntimeError("Model not trained.")
        H = self.encode(coords)
        return H @ self.W_out.T

    def predict_clipped(self, coords: np.ndarray) -> np.ndarray:
        """Predict and clip to [0, 1]."""
        return np.clip(self.predict(coords), 0, 1)


class FourierFeaturesMLP:
    """
    Fourier Features with MLP (gradient descent training).

    Original paper approach - more flexible but slower training.
    """

    def __init__(
        self,
        embedding_size: int,
        hidden_layers: List[int] = [256, 256],
        input_dim: int = 2,
        output_dim: int = 3,
        sigma: float = 10.0,
        seed: int = 42
    ):
        """
        Initialize Fourier + MLP model.

        Args:
            embedding_size: Number of Fourier frequencies
            hidden_layers: List of hidden layer sizes
            input_dim: Input coordinate dimension
            output_dim: Output dimension (3 for RGB)
            sigma: Frequency scale
            seed: Random seed
        """
        self.embedding_size = embedding_size
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.encoding_dim = 2 * embedding_size

        np.random.seed(seed)

        # Fourier frequency matrix (fixed)
        self.B = np.random.randn(embedding_size, input_dim) * sigma

        # Initialize MLP weights
        self.weights = []
        self.biases = []

        layer_sizes = [self.encoding_dim] + hidden_layers + [output_dim]

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            # Xavier initialization
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            self.weights.append(W)
            self.biases.append(b)

        self._print_init_info()

    def _print_init_info(self):
        """Print initialization summary."""
        total_params = self.B.size
        for W, b in zip(self.weights, self.biases):
            total_params += W.size + b.size

        print(f"Fourier Features (MLP) initialized:")
        print(f"  Embedding size: {self.embedding_size}")
        print(f"  Encoding dimension: {self.encoding_dim}")
        print(f"  Frequency scale (σ): {self.sigma}")
        print(f"  MLP architecture: {[self.encoding_dim] + self.hidden_layers + [self.output_dim]}")
        print(f"  Total params: {total_params}")

    def encode(self, coords: np.ndarray) -> np.ndarray:
        """Apply Fourier encoding."""
        projection = 2 * np.pi * coords @ self.B.T
        return np.concatenate([np.sin(projection), np.cos(projection)], axis=1)

    def forward(self, coords: np.ndarray) -> np.ndarray:
        """Forward pass through encoder + MLP."""
        x = self.encode(coords)

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = x @ self.weights[i] + self.biases[i]
            x = np.maximum(0, x)  # ReLU

        # Output layer with sigmoid
        x = x @ self.weights[-1] + self.biases[-1]
        x = 1 / (1 + np.exp(-x))  # Sigmoid for [0, 1] output

        return x

    def fit(self, coords: np.ndarray, targets: np.ndarray,
            lr: float = 1e-3, iterations: int = 1000, batch_size: int = 4096):
        """Train with gradient descent."""
        print(f"\nTraining MLP with gradient descent...")
        print(f"  Learning rate: {lr}")
        print(f"  Iterations: {iterations}")
        print(f"  Batch size: {batch_size}")

        n_samples = coords.shape[0]

        for iteration in range(iterations):
            # Mini-batch
            idx = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            batch_coords = coords[idx]
            batch_targets = targets[idx]

            # Forward pass with cached activations
            activations = [self.encode(batch_coords)]

            for i in range(len(self.weights) - 1):
                z = activations[-1] @ self.weights[i] + self.biases[i]
                a = np.maximum(0, z)  # ReLU
                activations.append(a)

            # Output
            z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
            pred = 1 / (1 + np.exp(-z_out))  # Sigmoid

            # Loss (MSE)
            loss = np.mean((pred - batch_targets) ** 2)

            # Backward pass
            # Output gradient
            d_pred = 2 * (pred - batch_targets) / batch_targets.shape[0]
            d_sigmoid = pred * (1 - pred)
            delta = d_pred * d_sigmoid

            # Gradients for output layer
            dW_out = activations[-1].T @ delta
            db_out = np.sum(delta, axis=0)

            # Backprop through hidden layers
            dWs = [dW_out]
            dbs = [db_out]

            for i in range(len(self.weights) - 2, -1, -1):
                delta = (delta @ self.weights[i + 1].T) * (activations[i + 1] > 0)  # ReLU derivative
                dW = activations[i].T @ delta
                db = np.sum(delta, axis=0)
                dWs.insert(0, dW)
                dbs.insert(0, db)

            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= lr * dWs[i]
                self.biases[i] -= lr * dbs[i]

            if iteration % 200 == 0:
                full_pred = self.forward(coords)
                full_loss = np.mean((full_pred - targets) ** 2)
                psnr = 10 * np.log10(1.0 / full_loss) if full_loss > 0 else float('inf')
                print(f"  Iter {iteration}: Loss={full_loss:.6f}, PSNR={psnr:.2f} dB")

        print("  Training complete!")

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict output."""
        return self.forward(coords)

    def predict_clipped(self, coords: np.ndarray) -> np.ndarray:
        """Predict and clip."""
        return np.clip(self.predict(coords), 0, 1)


def create_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """Create synthetic test image."""
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


def study_sigma(image: np.ndarray, embedding_size: int = 256):
    """Study effect of frequency scale σ."""
    print("\n" + "=" * 70)
    print(f"Frequency Scale (σ) Study - Embedding size: {embedding_size}")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    sigmas = [1, 2, 5, 10, 15, 20, 30, 50]
    results = []

    for sigma in sigmas:
        model = FourierFeaturesRidge(
            embedding_size=embedding_size,
            sigma=sigma,
            lambda_reg=1e-6
        )
        model.fit(coords, targets)
        pred = model.predict_clipped(coords).reshape(height, width, 3)
        psnr = compute_psnr(image, pred)

        print(f"σ={sigma:3d}: PSNR={psnr:.2f} dB")
        results.append({'sigma': sigma, 'psnr': psnr, 'pred': pred})

    return results


def study_embedding_size(image: np.ndarray, sigma: float = 10.0):
    """Study effect of embedding dimension."""
    print("\n" + "=" * 70)
    print(f"Embedding Size Study - σ={sigma}")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    sizes = [32, 64, 128, 256, 384, 512]
    results = []

    for m in sizes:
        model = FourierFeaturesRidge(
            embedding_size=m,
            sigma=sigma,
            lambda_reg=1e-6
        )
        model.fit(coords, targets)
        pred = model.predict_clipped(coords).reshape(height, width, 3)
        psnr = compute_psnr(image, pred)

        encoding_dim = 2 * m
        print(f"m={m:3d} (encoding={encoding_dim:4d}): PSNR={psnr:.2f} dB")
        results.append({'m': m, 'encoding_dim': encoding_dim, 'psnr': psnr, 'pred': pred})

    return results


def compare_all_methods(image: np.ndarray):
    """
    Comprehensive comparison: Fourier vs Reservoir methods.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON: Fourier vs Reservoir")
    print("=" * 70)

    height, width, _ = image.shape
    coords = create_coordinate_grid(height, width)
    targets = image.reshape(-1, 3)

    results = {}

    # 1. Fourier + Ridge (σ=10, m=256 → encoding=512)
    print("\n--- Fourier + Ridge (σ=10, m=256) ---")
    ff_ridge = FourierFeaturesRidge(embedding_size=256, sigma=10.0, lambda_reg=1e-6)
    ff_ridge.fit(coords, targets)
    pred_ff_ridge = ff_ridge.predict_clipped(coords).reshape(height, width, 3)
    psnr_ff_ridge = compute_psnr(image, pred_ff_ridge)
    print(f"PSNR: {psnr_ff_ridge:.2f} dB")
    results['fourier_ridge_256'] = {'pred': pred_ff_ridge, 'psnr': psnr_ff_ridge,
                                     'params': 256 * 2 + 512 * 3}

    # 2. Fourier + Ridge (σ=10, m=384 → encoding=768, same as best reservoir)
    print("\n--- Fourier + Ridge (σ=10, m=384) ---")
    ff_ridge_384 = FourierFeaturesRidge(embedding_size=384, sigma=10.0, lambda_reg=1e-6)
    ff_ridge_384.fit(coords, targets)
    pred_ff_ridge_384 = ff_ridge_384.predict_clipped(coords).reshape(height, width, 3)
    psnr_ff_ridge_384 = compute_psnr(image, pred_ff_ridge_384)
    print(f"PSNR: {psnr_ff_ridge_384:.2f} dB")
    results['fourier_ridge_384'] = {'pred': pred_ff_ridge_384, 'psnr': psnr_ff_ridge_384}

    # 3. Fourier + MLP (original paper style)
    print("\n--- Fourier + MLP (σ=10, m=256, 2-layer MLP) ---")
    ff_mlp = FourierFeaturesMLP(embedding_size=256, hidden_layers=[256, 256], sigma=10.0)
    ff_mlp.fit(coords, targets, lr=1e-2, iterations=1000)
    pred_ff_mlp = ff_mlp.predict_clipped(coords).reshape(height, width, 3)
    psnr_ff_mlp = compute_psnr(image, pred_ff_mlp)
    print(f"Final PSNR: {psnr_ff_mlp:.2f} dB")
    results['fourier_mlp'] = {'pred': pred_ff_mlp, 'psnr': psnr_ff_mlp}

    # 4. Import and run reservoir methods for comparison
    # Task 3: Self-Recurrent Settling
    print("\n--- Task 3: Self-Recurrent Reservoir (H=512, K=10) ---")
    from task3_self_recurrent_settling import SelfRecurrentReservoirINR
    res_single = SelfRecurrentReservoirINR(
        hidden_size=512, n_iterations=10, spectral_radius=0.9, lambda_reg=1e-6
    )
    res_single.fit(coords, targets)
    pred_res_single = res_single.predict_clipped(coords).reshape(height, width, 3)
    psnr_res_single = compute_psnr(image, pred_res_single)
    print(f"PSNR: {psnr_res_single:.2f} dB")
    results['reservoir_single'] = {'pred': pred_res_single, 'psnr': psnr_res_single}

    # 5. Task 5: Multi-Reservoir Ensemble
    print("\n--- Task 5: Multi-Reservoir (H=256x3, SR=[0.5,0.9,0.99]) ---")
    from task5_multi_reservoir_ensemble import MultiReservoirEnsemble
    res_multi = MultiReservoirEnsemble(
        hidden_size_per_reservoir=256,
        spectral_radii=[0.5, 0.9, 0.99],
        n_iterations=10,
        lambda_reg=1e-6
    )
    res_multi.fit(coords, targets)
    pred_res_multi = res_multi.predict_clipped(coords).reshape(height, width, 3)
    psnr_res_multi = compute_psnr(image, pred_res_multi)
    print(f"PSNR: {psnr_res_multi:.2f} dB")
    results['reservoir_multi'] = {'pred': pred_res_multi, 'psnr': psnr_res_multi}

    # 6. No encoding baseline (just ridge on raw coordinates)
    print("\n--- No Encoding Baseline (raw coords → ridge) ---")
    HTH = coords.T @ coords + 1e-6 * np.eye(2)
    HTY = coords.T @ targets
    W_out = np.linalg.solve(HTH, HTY)
    pred_none = np.clip(coords @ W_out, 0, 1).reshape(height, width, 3)
    psnr_none = compute_psnr(image, pred_none)
    print(f"PSNR: {psnr_none:.2f} dB")
    results['no_encoding'] = {'pred': pred_none, 'psnr': psnr_none}

    return results


def visualize_comparison(image: np.ndarray, results: dict, save_path: str = None):
    """Visualize all methods comparison."""
    n_methods = len(results)
    fig, axes = plt.subplots(2, (n_methods + 2) // 2, figsize=(4 * ((n_methods + 2) // 2), 8))
    axes = axes.flatten()

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Results
    for i, (name, data) in enumerate(results.items()):
        ax = axes[i + 1]
        ax.imshow(data['pred'])
        ax.set_title(f'{name}\n{data["psnr"]:.1f} dB')
        ax.axis('off')

    # Hide unused
    for i in range(len(results) + 1, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


if __name__ == "__main__":
    # Create test image
    print("Creating test image...")
    image = create_test_image(height=64, width=64)
    print(f"Image shape: {image.shape}")

    # Study sigma effect
    sigma_results = study_sigma(image, embedding_size=256)

    # Find best sigma
    best_sigma_result = max(sigma_results, key=lambda x: x['psnr'])
    print(f"\nBest σ: {best_sigma_result['sigma']} with PSNR={best_sigma_result['psnr']:.2f} dB")

    # Study embedding size
    embed_results = study_embedding_size(image, sigma=best_sigma_result['sigma'])

    # Comprehensive comparison
    comparison = compare_all_methods(image)

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'PSNR (dB)':<12}")
    print("-" * 47)

    sorted_results = sorted(comparison.items(), key=lambda x: x[1]['psnr'], reverse=True)
    for name, data in sorted_results:
        print(f"{name:<35} {data['psnr']:<12.2f}")

    # Visualize
    visualize_comparison(
        image, comparison,
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/fourier_vs_reservoir.png"
    )

    # Visualize sigma study
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, res in enumerate(sigma_results[:7]):
        axes[i + 1].imshow(res['pred'])
        axes[i + 1].set_title(f"σ={res['sigma']}\n{res['psnr']:.1f} dB")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig("/Users/davidpark/Documents/Claude/ReservoirINR/fourier_sigma_study.png",
                dpi=150, bbox_inches='tight')
    print("\nSigma study figure saved.")
    plt.close()

    print("\nFourier Features baseline complete.")

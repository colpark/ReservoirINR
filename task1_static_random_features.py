"""
Task 1: Static Random Features for INR (Baseline)

This is NOT a true reservoir - no recurrence.
It's an Extreme Learning Machine / Random Kitchen Sinks approach.

Architecture:
    Input: [x, y] coordinate (2D)
           ↓
    h = tanh(W_in · [x,y] + b)    # Fixed random projection
           ↓                       # Shape: (hidden_size,)
    output = W_out · h             # Linear readout (ridge regression)
           ↓
    RGB value (3D)

Components:
    - W_in: Fixed random matrix, shape (hidden_size, 2), drawn from N(0, 1/sqrt(2))
    - b: Fixed random bias, shape (hidden_size,), drawn from U(-1, 1)
    - W_out: ONLY trainable part, shape (3, hidden_size), trained via ridge regression

Training:
    Ridge regression closed-form solution:
    W_out = Y^T H (H^T H + λI)^{-1}

    Where:
    - H: design matrix of hidden states, shape (n_pixels, hidden_size)
    - Y: target RGB values, shape (n_pixels, 3)
    - λ: regularization parameter
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class StaticRandomFeatureINR:
    """
    Static Random Feature model for Implicit Neural Representation.

    This uses a single layer of random features (no recurrence).
    Equivalent to Extreme Learning Machine.
    """

    def __init__(self, hidden_size: int, input_dim: int = 2, output_dim: int = 3,
                 lambda_reg: float = 1e-6, seed: int = 42):
        """
        Initialize the model with fixed random weights.

        Args:
            hidden_size: Number of random features (basis functions)
            input_dim: Dimension of input coordinates (2 for images)
            output_dim: Dimension of output (3 for RGB)
            lambda_reg: Ridge regression regularization parameter
            seed: Random seed for reproducibility
        """
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_reg = lambda_reg

        # Set random seed for reproducibility
        np.random.seed(seed)

        # ============================================================
        # FIXED RANDOM WEIGHTS (never trained)
        # ============================================================

        # W_in: Input-to-hidden weights
        # Shape: (hidden_size, input_dim) = (hidden_size, 2)
        # Initialization: N(0, 1/sqrt(input_dim)) for reasonable activation range
        self.W_in = np.random.randn(hidden_size, input_dim) / np.sqrt(input_dim)

        # b: Bias vector
        # Shape: (hidden_size,)
        # Initialization: U(-1, 1) to spread activations
        self.b = np.random.uniform(-1, 1, size=(hidden_size,))

        # ============================================================
        # TRAINABLE WEIGHTS (trained via ridge regression)
        # ============================================================

        # W_out: Hidden-to-output weights
        # Shape: (output_dim, hidden_size) = (3, hidden_size)
        # Initialized to None, set during training
        self.W_out = None

        print(f"Model initialized:")
        print(f"  W_in shape: {self.W_in.shape} (fixed)")
        print(f"  b shape: {self.b.shape} (fixed)")
        print(f"  W_out shape: ({output_dim}, {hidden_size}) (trainable)")
        print(f"  Total fixed params: {self.W_in.size + self.b.size}")
        print(f"  Total trainable params: {output_dim * hidden_size}")

    def compute_hidden(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute hidden layer activations for given coordinates.

        Operation:
            h = tanh(W_in @ coords.T + b)

        Args:
            coords: Input coordinates, shape (n_samples, 2)

        Returns:
            Hidden activations, shape (n_samples, hidden_size)
        """
        # coords: (n_samples, 2)
        # W_in: (hidden_size, 2)
        # W_in @ coords.T: (hidden_size, n_samples)

        # Linear transformation
        pre_activation = self.W_in @ coords.T  # (hidden_size, n_samples)
        pre_activation = pre_activation + self.b[:, np.newaxis]  # Add bias (broadcasting)

        # Nonlinearity: tanh
        # This is the ONLY nonlinearity in the model
        h = np.tanh(pre_activation)  # (hidden_size, n_samples)

        return h.T  # Return (n_samples, hidden_size)

    def fit(self, coords: np.ndarray, targets: np.ndarray):
        """
        Train the output weights using ridge regression.

        Closed-form solution:
            W_out = Y^T @ H @ (H^T @ H + λI)^{-1}

        Or equivalently solving:
            (H^T @ H + λI) @ W_out^T = H^T @ Y

        Args:
            coords: Input coordinates, shape (n_samples, 2)
            targets: Target RGB values, shape (n_samples, 3)
        """
        print(f"\nTraining...")
        print(f"  Input coords shape: {coords.shape}")
        print(f"  Target values shape: {targets.shape}")

        # Step 1: Compute hidden activations for all coordinates
        H = self.compute_hidden(coords)  # (n_samples, hidden_size)
        print(f"  Hidden activations shape: {H.shape}")

        # Step 2: Ridge regression closed-form solution
        # We want to solve: min_W ||H @ W^T - Y||^2 + λ||W||^2
        # Solution: W^T = (H^T H + λI)^{-1} H^T Y

        # H^T @ H: (hidden_size, hidden_size)
        HTH = H.T @ H
        print(f"  H^T @ H shape: {HTH.shape}")

        # Regularization: add λI to diagonal
        reg_matrix = HTH + self.lambda_reg * np.eye(self.hidden_size)

        # H^T @ Y: (hidden_size, output_dim)
        HTY = H.T @ targets
        print(f"  H^T @ Y shape: {HTY.shape}")

        # Solve linear system: reg_matrix @ W_out^T = HTY
        # Using np.linalg.solve for numerical stability
        W_out_T = np.linalg.solve(reg_matrix, HTY)  # (hidden_size, output_dim)

        self.W_out = W_out_T.T  # (output_dim, hidden_size)
        print(f"  W_out shape: {self.W_out.shape}")
        print(f"  Training complete!")

    def predict(self, coords: np.ndarray) -> np.ndarray:
        """
        Predict RGB values for given coordinates.

        Operation:
            output = W_out @ h

        Args:
            coords: Input coordinates, shape (n_samples, 2)

        Returns:
            Predicted RGB values, shape (n_samples, 3)
        """
        if self.W_out is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Compute hidden activations
        H = self.compute_hidden(coords)  # (n_samples, hidden_size)

        # Linear output: H @ W_out^T
        output = H @ self.W_out.T  # (n_samples, output_dim)

        return output

    def predict_clipped(self, coords: np.ndarray) -> np.ndarray:
        """Predict and clip to valid RGB range [0, 1]."""
        return np.clip(self.predict(coords), 0, 1)


def create_coordinate_grid(height: int, width: int) -> np.ndarray:
    """
    Create normalized coordinate grid for an image.

    Coordinates are normalized to [-1, 1] range.

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Coordinate array, shape (height * width, 2)
        Each row is [x, y] where x, y ∈ [-1, 1]
    """
    # Create 1D coordinate arrays
    x = np.linspace(-1, 1, width)   # Horizontal (column) coordinate
    y = np.linspace(-1, 1, height)  # Vertical (row) coordinate

    # Create 2D meshgrid
    xx, yy = np.meshgrid(x, y)

    # Flatten and stack: shape (H*W, 2)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    return coords


def load_image(path: str, max_size: int = 256) -> np.ndarray:
    """
    Load and preprocess image.

    Args:
        path: Path to image file
        max_size: Maximum dimension (for memory efficiency)

    Returns:
        Image array, shape (H, W, 3), values in [0, 1]
    """
    img = Image.open(path).convert('RGB')

    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img) / 255.0

    return img_array


def create_test_image(height: int = 64, width: int = 64) -> np.ndarray:
    """
    Create a simple synthetic test image with various frequencies.

    Contains:
    - Low frequency: smooth gradient
    - Medium frequency: sinusoidal pattern
    - High frequency: checkerboard in corner
    """
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Low frequency: diagonal gradient
    low_freq = (xx + yy) / 2

    # Medium frequency: sinusoidal
    med_freq = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * xx) * np.sin(2 * np.pi * 4 * yy)

    # Combine into RGB
    img = np.stack([
        low_freq,                    # R: gradient
        med_freq,                    # G: sinusoidal
        0.5 * np.ones_like(xx)       # B: constant
    ], axis=-1)

    return img


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    return np.mean((img1 - img2) ** 2)


def run_experiment(image: np.ndarray, hidden_sizes: list, lambda_reg: float = 1e-6):
    """
    Run experiment with different hidden sizes.

    Args:
        image: Target image, shape (H, W, 3)
        hidden_sizes: List of hidden layer sizes to test
        lambda_reg: Ridge regression regularization

    Returns:
        Dictionary with results
    """
    height, width, _ = image.shape
    n_pixels = height * width

    print(f"Image size: {height} x {width} = {n_pixels} pixels")
    print(f"Testing hidden sizes: {hidden_sizes}")
    print("=" * 60)

    # Create coordinate grid
    coords = create_coordinate_grid(height, width)  # (n_pixels, 2)

    # Flatten image to target values
    targets = image.reshape(-1, 3)  # (n_pixels, 3)

    results = {
        'hidden_sizes': hidden_sizes,
        'psnr': [],
        'mse': [],
        'predictions': []
    }

    for hidden_size in hidden_sizes:
        print(f"\n{'='*60}")
        print(f"Hidden size: {hidden_size}")
        print(f"{'='*60}")

        # Create and train model
        model = StaticRandomFeatureINR(
            hidden_size=hidden_size,
            input_dim=2,
            output_dim=3,
            lambda_reg=lambda_reg
        )

        model.fit(coords, targets)

        # Predict
        predictions = model.predict_clipped(coords)
        pred_image = predictions.reshape(height, width, 3)

        # Compute metrics
        psnr = compute_psnr(image, pred_image)
        mse = compute_mse(image, pred_image)

        print(f"\nResults:")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")

        results['psnr'].append(psnr)
        results['mse'].append(mse)
        results['predictions'].append(pred_image)

    return results


def visualize_results(image: np.ndarray, results: dict, save_path: str = None):
    """Visualize original image and reconstructions."""
    n_models = len(results['hidden_sizes'])

    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Reconstructions
    for i, (hidden_size, pred, psnr) in enumerate(zip(
        results['hidden_sizes'],
        results['predictions'],
        results['psnr']
    )):
        axes[i + 1].imshow(pred)
        axes[i + 1].set_title(f'H={hidden_size}\nPSNR={psnr:.1f}dB')
        axes[i + 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    print("Task 1: Static Random Features for INR")
    print("=" * 60)
    print("\nThis is a BASELINE - not a true reservoir (no recurrence).")
    print("Architecture: coords → tanh(W_in @ coords + b) → W_out → RGB")
    print("Only W_out is trained (via ridge regression).")
    print("=" * 60)

    # Create synthetic test image
    print("\nCreating synthetic test image...")
    image = create_test_image(height=64, width=64)
    print(f"Image shape: {image.shape}")

    # Test different hidden sizes
    hidden_sizes = [64, 256, 1024, 4096]

    # Run experiment
    results = run_experiment(
        image=image,
        hidden_sizes=hidden_sizes,
        lambda_reg=1e-6
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Hidden Size':<15} {'PSNR (dB)':<15} {'MSE':<15}")
    print("-" * 45)
    for h, psnr, mse in zip(results['hidden_sizes'], results['psnr'], results['mse']):
        print(f"{h:<15} {psnr:<15.2f} {mse:<15.6f}")

    # Visualize
    visualize_results(
        image,
        results,
        save_path="/Users/davidpark/Documents/Claude/ReservoirINR/task1_results.png"
    )

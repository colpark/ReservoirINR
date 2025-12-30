"""
Real Image INR Comparison: Fourier vs Deep Reservoir

Testing on fig/cat.png - a real image with complex spatial structure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(42)

# =============================================================================
# LOAD IMAGE
# =============================================================================

print("=" * 70)
print("REAL IMAGE INR: Fourier vs Deep Reservoir")
print("=" * 70)

# Load and preprocess image
img = Image.open('fig/cat.png').convert('RGB')
print(f"Original image size: {img.size}")

# Resize for manageable computation
target_size = 128
img = img.resize((target_size, target_size), Image.LANCZOS)
img_array = np.array(img) / 255.0  # Normalize to [0, 1]

print(f"Resized to: {img_array.shape}")

# Create coordinate grid
h, w, c = img_array.shape
y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)  # (H*W, 2)
pixels = img_array.reshape(-1, 3)  # (H*W, 3) - RGB values

print(f"Total pixels: {len(pixels)}")

# =============================================================================
# FEATURE GENERATORS
# =============================================================================

def fourier_features(x, num_features, sigma):
    """Random Fourier features"""
    np.random.seed(42)
    B = np.random.randn(x.shape[1], num_features) * sigma
    proj = x @ B
    return np.concatenate([np.sin(2*np.pi*proj), np.cos(2*np.pi*proj)], axis=1)


def deep_reservoir(x, hidden_size, num_layers=5, iterations=10, spectral_radius=0.9):
    """Deep stacked reservoir"""
    np.random.seed(42)
    n, d = x.shape

    layers = []
    d_in = d
    for layer in range(num_layers):
        np.random.seed(42 + layer)
        W_in = np.random.randn(d_in, hidden_size) * 0.5
        W_hh = np.random.randn(hidden_size, hidden_size)
        eig = np.abs(np.linalg.eigvals(W_hh)).max()
        W_hh = W_hh * (spectral_radius / eig)
        b = np.random.randn(hidden_size) * 0.1
        layers.append((W_in, W_hh, b))
        d_in = hidden_size

    all_states = []
    for i in range(n):
        layer_input = x[i:i+1]
        layer_states = []

        for W_in, W_hh, b in layers:
            h = np.zeros(hidden_size)
            for _ in range(iterations):
                h = np.tanh(layer_input @ W_in + h @ W_hh + b).flatten()
            layer_states.append(h)
            layer_input = h.reshape(1, -1)

        all_states.append(np.concatenate(layer_states))

    return np.array(all_states)


def basic_reservoir(x, hidden_size, iterations=10, spectral_radius=0.9, leak=0.3):
    """Basic single-layer reservoir"""
    np.random.seed(42)
    n, d = x.shape

    W_in = np.random.randn(d, hidden_size) * 0.5
    W_hh = np.random.randn(hidden_size, hidden_size)
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    W_hh = W_hh * (spectral_radius / eig)
    b = np.random.randn(hidden_size) * 0.1

    H = np.zeros((n, hidden_size))
    for i in range(n):
        h = np.zeros(hidden_size)
        for _ in range(iterations):
            h = (1-leak) * h + leak * np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H


def ridge(H, y, lamb=1e-6):
    """Ridge regression"""
    return np.linalg.solve(H.T @ H + lamb * np.eye(H.shape[1]), H.T @ y)


def compute_metrics(pred, target):
    """Compute MSE and PSNR"""
    mse = np.mean((pred - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
    return mse, psnr


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

print("\nTesting different configurations...")

results = {}

# 1. Fourier Features (various sigma)
print("\n--- FOURIER FEATURES ---")
for sigma in [1.0, 2.0, 5.0, 10.0]:
    for num_feat in [128, 256]:
        name = f"fourier_σ{sigma}_f{num_feat}"
        H = fourier_features(coords, num_feat, sigma)
        W = ridge(H, pixels)
        pred = np.clip(H @ W, 0, 1)
        mse, psnr = compute_metrics(pred, pixels)
        results[name] = {'mse': mse, 'psnr': psnr, 'pred': pred, 'dim': H.shape[1]}
        print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# 2. Basic Reservoir
print("\n--- BASIC RESERVOIR ---")
for hidden in [256, 512, 1024]:
    name = f"basic_res_{hidden}"
    H = basic_reservoir(coords, hidden, iterations=10)
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    mse, psnr = compute_metrics(pred, pixels)
    results[name] = {'mse': mse, 'psnr': psnr, 'pred': pred, 'dim': H.shape[1]}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# 3. Deep Reservoir
print("\n--- DEEP RESERVOIR ---")
for num_layers in [3, 5, 7]:
    for hidden in [64, 128]:
        name = f"deep_res_L{num_layers}_H{hidden}"
        H = deep_reservoir(coords, hidden, num_layers=num_layers, iterations=10)
        W = ridge(H, pixels)
        pred = np.clip(H @ W, 0, 1)
        mse, psnr = compute_metrics(pred, pixels)
        results[name] = {'mse': mse, 'psnr': psnr, 'pred': pred, 'dim': H.shape[1]}
        print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# 4. Very Deep Reservoir
print("\n--- VERY DEEP RESERVOIR ---")
for num_layers in [10, 15]:
    name = f"vdeep_res_L{num_layers}_H64"
    H = deep_reservoir(coords, 64, num_layers=num_layers, iterations=5)
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    mse, psnr = compute_metrics(pred, pixels)
    results[name] = {'mse': mse, 'psnr': psnr, 'pred': pred, 'dim': H.shape[1]}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# =============================================================================
# FIND BEST
# =============================================================================

print("\n" + "=" * 70)
print("BEST RESULTS")
print("=" * 70)

# Best Fourier
fourier_results = {k: v for k, v in results.items() if 'fourier' in k}
best_fourier = max(fourier_results.keys(), key=lambda k: fourier_results[k]['psnr'])
print(f"\nBest Fourier: {best_fourier}")
print(f"  PSNR = {results[best_fourier]['psnr']:.2f} dB")

# Best Reservoir (basic)
basic_results = {k: v for k, v in results.items() if 'basic' in k}
best_basic = max(basic_results.keys(), key=lambda k: basic_results[k]['psnr'])
print(f"\nBest Basic Reservoir: {best_basic}")
print(f"  PSNR = {results[best_basic]['psnr']:.2f} dB")

# Best Deep Reservoir
deep_results = {k: v for k, v in results.items() if 'deep' in k or 'vdeep' in k}
best_deep = max(deep_results.keys(), key=lambda k: deep_results[k]['psnr'])
print(f"\nBest Deep Reservoir: {best_deep}")
print(f"  PSNR = {results[best_deep]['psnr']:.2f} dB")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Images
axes[0, 0].imshow(img_array)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# Best Fourier reconstruction
pred_fourier = results[best_fourier]['pred'].reshape(h, w, 3)
axes[0, 1].imshow(pred_fourier)
axes[0, 1].set_title(f'Fourier\nPSNR={results[best_fourier]["psnr"]:.1f} dB')
axes[0, 1].axis('off')

# Best Basic Reservoir reconstruction
pred_basic = results[best_basic]['pred'].reshape(h, w, 3)
axes[0, 2].imshow(pred_basic)
axes[0, 2].set_title(f'Basic Reservoir\nPSNR={results[best_basic]["psnr"]:.1f} dB')
axes[0, 2].axis('off')

# Best Deep Reservoir reconstruction
pred_deep = results[best_deep]['pred'].reshape(h, w, 3)
axes[0, 3].imshow(pred_deep)
axes[0, 3].set_title(f'Deep Reservoir\nPSNR={results[best_deep]["psnr"]:.1f} dB')
axes[0, 3].axis('off')

# Row 2: Error maps
axes[1, 0].axis('off')

error_fourier = np.mean(np.abs(pred_fourier - img_array), axis=2)
axes[1, 1].imshow(error_fourier, cmap='hot', vmin=0, vmax=0.2)
axes[1, 1].set_title(f'Fourier Error\nMSE={results[best_fourier]["mse"]:.4f}')
axes[1, 1].axis('off')

error_basic = np.mean(np.abs(pred_basic - img_array), axis=2)
axes[1, 2].imshow(error_basic, cmap='hot', vmin=0, vmax=0.2)
axes[1, 2].set_title(f'Basic Res. Error\nMSE={results[best_basic]["mse"]:.4f}')
axes[1, 2].axis('off')

error_deep = np.mean(np.abs(pred_deep - img_array), axis=2)
axes[1, 3].imshow(error_deep, cmap='hot', vmin=0, vmax=0.2)
axes[1, 3].set_title(f'Deep Res. Error\nMSE={results[best_deep]["mse"]:.4f}')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('cat_inr_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar chart of all results
fig, ax = plt.subplots(figsize=(14, 6))

sorted_results = sorted(results.items(), key=lambda x: x[1]['psnr'], reverse=True)
names = [r[0] for r in sorted_results]
psnrs = [r[1]['psnr'] for r in sorted_results]
colors = ['blue' if 'fourier' in n else ('green' if 'deep' in n or 'vdeep' in n else 'red') for n in names]

bars = ax.bar(range(len(names)), psnrs, color=colors, alpha=0.7)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('PSNR (dB)')
ax.set_title('Cat Image INR: Fourier vs Reservoir Comparison')
ax.grid(True, alpha=0.3, axis='y')

# Legend
ax.bar([], [], color='blue', alpha=0.7, label='Fourier')
ax.bar([], [], color='green', alpha=0.7, label='Deep Reservoir')
ax.bar([], [], color='red', alpha=0.7, label='Basic Reservoir')
ax.legend()

plt.tight_layout()
plt.savefig('cat_inr_barchart.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved to: cat_inr_comparison.png, cat_inr_barchart.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Real Image (Cat) INR Results")
print("=" * 70)

print(f"""
Image: {target_size}x{target_size} RGB ({target_size**2 * 3} values to predict)

┌────────────────────────────────────────────────────────────────────┐
│                         PSNR COMPARISON                            │
├────────────────────────────────────────────────────────────────────┤
│  Best Fourier:         {results[best_fourier]['psnr']:>6.2f} dB  ({best_fourier})
│  Best Basic Reservoir: {results[best_basic]['psnr']:>6.2f} dB  ({best_basic})
│  Best Deep Reservoir:  {results[best_deep]['psnr']:>6.2f} dB  ({best_deep})
├────────────────────────────────────────────────────────────────────┤
""")

gap_basic = results[best_fourier]['psnr'] - results[best_basic]['psnr']
gap_deep = results[best_fourier]['psnr'] - results[best_deep]['psnr']

print(f"│  Gap (Fourier - Basic):  {gap_basic:>+6.2f} dB")
print(f"│  Gap (Fourier - Deep):   {gap_deep:>+6.2f} dB")
print("└────────────────────────────────────────────────────────────────────┘")

if gap_deep < 1.0:
    print("\n✓ Deep Reservoir is COMPETITIVE with Fourier on real images!")
elif gap_deep < 3.0:
    print("\n~ Deep Reservoir is CLOSE to Fourier on real images.")
else:
    print("\n✗ Fourier still significantly better on real images.")

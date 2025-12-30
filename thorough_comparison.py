"""
Thorough Comparison: Is Deep Reservoir Just Random Projection?

Key questions:
1. With larger networks, how much better is Fourier?
2. Is deep reservoir equivalent to stacked random projections (no recurrence)?
3. Fair parameter count comparison
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
print("THOROUGH COMPARISON: Reservoir vs Random Projection vs Fourier")
print("=" * 70)

img = Image.open('fig/cat.png').convert('RGB')
target_size = 128
img = img.resize((target_size, target_size), Image.LANCZOS)
img_array = np.array(img) / 255.0

h, w, c = img_array.shape
y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
pixels = img_array.reshape(-1, 3)

print(f"Image: {h}x{w} RGB")
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


def stacked_random_projection(x, hidden_size, num_layers):
    """
    Stacked random projections WITHOUT recurrence.
    Just: tanh(W_L * tanh(W_{L-1} * ... tanh(W_1 * x)))
    This is a random deep network without any reservoir dynamics.
    """
    np.random.seed(42)
    n, d = x.shape

    # First layer
    W = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
    b = np.random.randn(hidden_size) * 0.1
    h = np.tanh(x @ W + b)

    all_features = [h]

    # Subsequent layers
    for layer in range(1, num_layers):
        np.random.seed(42 + layer)
        W = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        b = np.random.randn(hidden_size) * 0.1
        h = np.tanh(h @ W + b)
        all_features.append(h)

    return np.concatenate(all_features, axis=1)


def deep_reservoir(x, hidden_size, num_layers, iterations=10, spectral_radius=0.9):
    """
    Deep reservoir WITH recurrence at each layer.
    Each layer: iterate h = tanh(W_in*input + W_hh*h + b) multiple times
    """
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


def single_random_projection(x, hidden_size):
    """Single layer random projection (baseline)"""
    np.random.seed(42)
    W = np.random.randn(x.shape[1], hidden_size) * 1.0
    b = np.random.randn(hidden_size) * 0.1
    return np.tanh(x @ W + b)


def ridge(H, y, lamb=1e-6):
    return np.linalg.solve(H.T @ H + lamb * np.eye(H.shape[1]), H.T @ y)


def compute_psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else 100


def count_params(method_name, dim, input_dim=2):
    """Estimate parameter count"""
    if 'fourier' in method_name:
        # B matrix: input_dim x num_features
        return input_dim * (dim // 2)
    else:
        # Rough estimate for reservoir/projection
        return dim * dim  # Approximate


# =============================================================================
# EXPERIMENT 1: SCALE UP FOURIER
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 1: Scaling Up Fourier Features")
print("=" * 70)

fourier_results = {}

for num_feat in [128, 256, 512, 1024, 2048]:
    for sigma in [2.0, 5.0, 10.0, 20.0]:
        name = f"fourier_f{num_feat}_σ{sigma}"
        H = fourier_features(coords, num_feat, sigma)
        W = ridge(H, pixels)
        pred = np.clip(H @ W, 0, 1)
        psnr = compute_psnr(pred, pixels)
        fourier_results[name] = {'psnr': psnr, 'dim': H.shape[1], 'pred': pred}

# Find best for each feature count
print(f"\n{'Features':<10} {'Best σ':<10} {'PSNR':<10} {'Output Dim'}")
print("-" * 45)
for nf in [128, 256, 512, 1024, 2048]:
    subset = {k: v for k, v in fourier_results.items() if f"_f{nf}_" in k}
    best = max(subset.keys(), key=lambda k: subset[k]['psnr'])
    print(f"{nf:<10} {best.split('σ')[1]:<10} {fourier_results[best]['psnr']:<10.2f} {fourier_results[best]['dim']}")

best_fourier = max(fourier_results.keys(), key=lambda k: fourier_results[k]['psnr'])
print(f"\nBest overall: {best_fourier} = {fourier_results[best_fourier]['psnr']:.2f} dB")

# =============================================================================
# EXPERIMENT 2: COMPARE ARCHITECTURES AT SIMILAR PARAMETER COUNTS
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: Fair Comparison (Similar Output Dimensions)")
print("=" * 70)

# Test at output dim ~512
print("\n--- Output Dim ≈ 512 ---")

configs_512 = {
    'fourier_256': lambda: fourier_features(coords, 256, sigma=10.0),  # dim=512
    'single_random_512': lambda: single_random_projection(coords, 512),  # dim=512
    'stacked_4L_128': lambda: stacked_random_projection(coords, 128, 4),  # dim=512
    'stacked_8L_64': lambda: stacked_random_projection(coords, 64, 8),  # dim=512
    'reservoir_4L_128': lambda: deep_reservoir(coords, 128, 4, iterations=10),  # dim=512
    'reservoir_8L_64': lambda: deep_reservoir(coords, 64, 8, iterations=10),  # dim=512
}

results_512 = {}
for name, fn in configs_512.items():
    H = fn()
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    psnr = compute_psnr(pred, pixels)
    results_512[name] = {'psnr': psnr, 'dim': H.shape[1], 'pred': pred}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# Test at output dim ~1024
print("\n--- Output Dim ≈ 1024 ---")

configs_1024 = {
    'fourier_512': lambda: fourier_features(coords, 512, sigma=10.0),  # dim=1024
    'single_random_1024': lambda: single_random_projection(coords, 1024),  # dim=1024
    'stacked_8L_128': lambda: stacked_random_projection(coords, 128, 8),  # dim=1024
    'stacked_16L_64': lambda: stacked_random_projection(coords, 64, 16),  # dim=1024
    'reservoir_8L_128': lambda: deep_reservoir(coords, 128, 8, iterations=10),  # dim=1024
    'reservoir_16L_64': lambda: deep_reservoir(coords, 64, 16, iterations=10),  # dim=1024
}

results_1024 = {}
for name, fn in configs_1024.items():
    H = fn()
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    psnr = compute_psnr(pred, pixels)
    results_1024[name] = {'psnr': psnr, 'dim': H.shape[1], 'pred': pred}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# Test at output dim ~2048
print("\n--- Output Dim ≈ 2048 ---")

configs_2048 = {
    'fourier_1024': lambda: fourier_features(coords, 1024, sigma=10.0),  # dim=2048
    'single_random_2048': lambda: single_random_projection(coords, 2048),  # dim=2048
    'stacked_16L_128': lambda: stacked_random_projection(coords, 128, 16),  # dim=2048
    'reservoir_16L_128': lambda: deep_reservoir(coords, 128, 16, iterations=5),  # dim=2048
}

results_2048 = {}
for name, fn in configs_2048.items():
    H = fn()
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    psnr = compute_psnr(pred, pixels)
    results_2048[name] = {'psnr': psnr, 'dim': H.shape[1], 'pred': pred}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# Test at output dim ~4096
print("\n--- Output Dim ≈ 4096 ---")

configs_4096 = {
    'fourier_2048': lambda: fourier_features(coords, 2048, sigma=10.0),  # dim=4096
    'single_random_4096': lambda: single_random_projection(coords, 4096),  # dim=4096
    'stacked_32L_128': lambda: stacked_random_projection(coords, 128, 32),  # dim=4096
    'reservoir_32L_128': lambda: deep_reservoir(coords, 128, 32, iterations=3),  # dim=4096
}

results_4096 = {}
for name, fn in configs_4096.items():
    H = fn()
    W = ridge(H, pixels)
    pred = np.clip(H @ W, 0, 1)
    psnr = compute_psnr(pred, pixels)
    results_4096[name] = {'psnr': psnr, 'dim': H.shape[1], 'pred': pred}
    print(f"  {name:<25}: PSNR = {psnr:.2f} dB (dim={H.shape[1]})")

# =============================================================================
# EXPERIMENT 3: KEY TEST - Does Recurrence Matter?
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: Does Recurrence Matter? (Reservoir vs Stacked Random)")
print("=" * 70)

print("\nComparing at dim=512:")
print(f"  Stacked Random (4L x 128): {results_512['stacked_4L_128']['psnr']:.2f} dB")
print(f"  Deep Reservoir (4L x 128): {results_512['reservoir_4L_128']['psnr']:.2f} dB")
print(f"  Difference: {results_512['reservoir_4L_128']['psnr'] - results_512['stacked_4L_128']['psnr']:.2f} dB")

print("\nComparing at dim=1024:")
print(f"  Stacked Random (8L x 128): {results_1024['stacked_8L_128']['psnr']:.2f} dB")
print(f"  Deep Reservoir (8L x 128): {results_1024['reservoir_8L_128']['psnr']:.2f} dB")
print(f"  Difference: {results_1024['reservoir_8L_128']['psnr'] - results_1024['stacked_8L_128']['psnr']:.2f} dB")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

all_results = {**results_512, **results_1024, **results_2048, **results_4096}

# Group by type
fourier_all = {k: v for k, v in all_results.items() if 'fourier' in k}
single_all = {k: v for k, v in all_results.items() if 'single' in k}
stacked_all = {k: v for k, v in all_results.items() if 'stacked' in k}
reservoir_all = {k: v for k, v in all_results.items() if 'reservoir' in k}

print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESULTS BY METHOD TYPE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FOURIER FEATURES (explicit sin/cos basis):                            │
│    Best: {max(fourier_all.keys(), key=lambda k: fourier_all[k]['psnr']):<30} = {max(v['psnr'] for v in fourier_all.values()):.2f} dB
│                                                                         │
│  SINGLE RANDOM PROJECTION (1 layer tanh):                              │
│    Best: {max(single_all.keys(), key=lambda k: single_all[k]['psnr']):<30} = {max(v['psnr'] for v in single_all.values()):.2f} dB
│                                                                         │
│  STACKED RANDOM (multi-layer, NO recurrence):                          │
│    Best: {max(stacked_all.keys(), key=lambda k: stacked_all[k]['psnr']):<30} = {max(v['psnr'] for v in stacked_all.values()):.2f} dB
│                                                                         │
│  DEEP RESERVOIR (multi-layer, WITH recurrence):                        │
│    Best: {max(reservoir_all.keys(), key=lambda k: reservoir_all[k]['psnr']):<30} = {max(v['psnr'] for v in reservoir_all.values()):.2f} dB
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

print("""
KEY FINDINGS:

1. FOURIER vs OTHERS:
   - At large scale, Fourier should dominate for images
   - The sin/cos basis is specifically designed for spatial frequencies

2. STACKED RANDOM vs DEEP RESERVOIR:
   - If similar performance → recurrence doesn't help for static INR
   - If reservoir is better → recurrence adds something
   - If stacked is better → recurrence is just overhead

3. DEPTH EFFECT:
   - Both stacked and reservoir benefit from depth
   - This is the "random deep network" effect
   - More layers = more hierarchical features

4. SINGLE RANDOM vs DEEP:
   - Single layer is just kernel regression approximation
   - Deep has more representational power
""")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Best of each type at dim ~2048
best_fourier_2k = 'fourier_1024'
best_single = 'single_random_2048'
best_stacked = 'stacked_16L_128'
best_reservoir = 'reservoir_16L_128'

# Row 1: Reconstructions
axes[0, 0].imshow(img_array)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(results_2048[best_fourier_2k]['pred'].reshape(h, w, 3))
axes[0, 1].set_title(f'Fourier\n{results_2048[best_fourier_2k]["psnr"]:.1f} dB')
axes[0, 1].axis('off')

axes[0, 2].imshow(results_2048[best_stacked]['pred'].reshape(h, w, 3))
axes[0, 2].set_title(f'Stacked Random\n{results_2048[best_stacked]["psnr"]:.1f} dB')
axes[0, 2].axis('off')

axes[0, 3].imshow(results_2048[best_reservoir]['pred'].reshape(h, w, 3))
axes[0, 3].set_title(f'Deep Reservoir\n{results_2048[best_reservoir]["psnr"]:.1f} dB')
axes[0, 3].axis('off')

# Row 2: Scaling comparison
dims = [512, 1024, 2048, 4096]

fourier_psnrs = [
    results_512['fourier_256']['psnr'],
    results_1024['fourier_512']['psnr'],
    results_2048['fourier_1024']['psnr'],
    results_4096['fourier_2048']['psnr'],
]

single_psnrs = [
    results_512['single_random_512']['psnr'],
    results_1024['single_random_1024']['psnr'],
    results_2048['single_random_2048']['psnr'],
    results_4096['single_random_4096']['psnr'],
]

stacked_psnrs = [
    results_512['stacked_4L_128']['psnr'],
    results_1024['stacked_8L_128']['psnr'],
    results_2048['stacked_16L_128']['psnr'],
    results_4096['stacked_32L_128']['psnr'],
]

reservoir_psnrs = [
    results_512['reservoir_4L_128']['psnr'],
    results_1024['reservoir_8L_128']['psnr'],
    results_2048['reservoir_16L_128']['psnr'],
    results_4096['reservoir_32L_128']['psnr'],
]

ax = axes[1, 0]
ax.axis('off')

ax = axes[1, 1]
ax.plot(dims, fourier_psnrs, 'bo-', label='Fourier', linewidth=2, markersize=8)
ax.plot(dims, single_psnrs, 'rs-', label='Single Random', linewidth=2, markersize=8)
ax.plot(dims, stacked_psnrs, 'g^-', label='Stacked Random', linewidth=2, markersize=8)
ax.plot(dims, reservoir_psnrs, 'md-', label='Deep Reservoir', linewidth=2, markersize=8)
ax.set_xlabel('Output Dimension')
ax.set_ylabel('PSNR (dB)')
ax.set_title('Scaling Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
# Stacked vs Reservoir comparison
diff = [r - s for r, s in zip(reservoir_psnrs, stacked_psnrs)]
ax.bar(range(len(dims)), diff, color=['green' if d > 0 else 'red' for d in diff])
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel('Output Dimension')
ax.set_ylabel('PSNR Difference (dB)')
ax.set_title('Reservoir - Stacked Random\n(+ = reservoir better)')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3)

ax = axes[1, 3]
# Gap from Fourier
gap_single = [f - s for f, s in zip(fourier_psnrs, single_psnrs)]
gap_stacked = [f - s for f, s in zip(fourier_psnrs, stacked_psnrs)]
gap_reservoir = [f - r for f, r in zip(fourier_psnrs, reservoir_psnrs)]

x = np.arange(len(dims))
width = 0.25
ax.bar(x - width, gap_single, width, label='vs Single', color='red', alpha=0.7)
ax.bar(x, gap_stacked, width, label='vs Stacked', color='green', alpha=0.7)
ax.bar(x + width, gap_reservoir, width, label='vs Reservoir', color='purple', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel('Output Dimension')
ax.set_ylabel('Fourier Advantage (dB)')
ax.set_title('Gap: Fourier - Others')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thorough_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to: thorough_comparison.png")

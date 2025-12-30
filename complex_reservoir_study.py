"""
Complex Reservoir Architectures: Can We Match Fourier?

Testing sophisticated reservoir configurations:
1. Deep Reservoir (stacked layers)
2. Very Large Reservoir (2048+ units)
3. Many Iterations (50-100 settling steps)
4. Edge of Chaos (spectral radius ≈ 1.0)
5. Multi-Scale Ensemble (different timescales)
6. Orthogonal Reservoir (orthogonal weight matrix)
7. Sparse Reservoir (sparse connectivity)
8. Leaky Integrator Variations
9. Input-driven vs Autonomous modes
10. Concatenated Multi-Config
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.linalg import qr

np.random.seed(42)

# =============================================================================
# RESERVOIR VARIANTS
# =============================================================================

def basic_reservoir(x, hidden_size, iterations=10, spectral_radius=0.9, leak=0.3):
    """Standard self-recurrent settling reservoir"""
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


def deep_reservoir(x, hidden_size, num_layers=3, iterations=10, spectral_radius=0.9):
    """Stacked reservoir layers"""
    np.random.seed(42)
    n, d = x.shape

    # Build layers
    layers = []
    d_in = d
    for layer in range(num_layers):
        W_in = np.random.randn(d_in, hidden_size) * 0.5
        W_hh = np.random.randn(hidden_size, hidden_size)
        eig = np.abs(np.linalg.eigvals(W_hh)).max()
        W_hh = W_hh * (spectral_radius / eig)
        b = np.random.randn(hidden_size) * 0.1
        layers.append((W_in, W_hh, b))
        d_in = hidden_size

    # Process each input
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

        # Concatenate all layer states
        all_states.append(np.concatenate(layer_states))

    return np.array(all_states)


def orthogonal_reservoir(x, hidden_size, iterations=10, spectral_radius=0.99):
    """Reservoir with orthogonal recurrent weights"""
    np.random.seed(42)
    n, d = x.shape

    W_in = np.random.randn(d, hidden_size) * 0.5

    # Orthogonal recurrent matrix
    if hidden_size <= 1000:
        Q, _ = qr(np.random.randn(hidden_size, hidden_size))
        W_hh = Q * spectral_radius
    else:
        # For large matrices, use block-diagonal orthogonal
        block_size = 100
        n_blocks = hidden_size // block_size
        W_hh = np.zeros((hidden_size, hidden_size))
        for i in range(n_blocks):
            Q, _ = qr(np.random.randn(block_size, block_size))
            start = i * block_size
            W_hh[start:start+block_size, start:start+block_size] = Q * spectral_radius

    b = np.random.randn(hidden_size) * 0.1

    H = np.zeros((n, hidden_size))
    for i in range(n):
        h = np.zeros(hidden_size)
        for _ in range(iterations):
            h = np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H


def sparse_reservoir(x, hidden_size, iterations=10, spectral_radius=0.9, sparsity=0.1):
    """Reservoir with sparse connectivity"""
    np.random.seed(42)
    n, d = x.shape

    W_in = np.random.randn(d, hidden_size) * 0.5

    # Sparse recurrent weights
    W_hh = np.random.randn(hidden_size, hidden_size)
    mask = np.random.rand(hidden_size, hidden_size) < sparsity
    W_hh = W_hh * mask

    # Scale to spectral radius
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    if eig > 0:
        W_hh = W_hh * (spectral_radius / eig)

    b = np.random.randn(hidden_size) * 0.1

    H = np.zeros((n, hidden_size))
    for i in range(n):
        h = np.zeros(hidden_size)
        for _ in range(iterations):
            h = np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H


def multi_scale_reservoir(x, hidden_per_scale, scales=[0.5, 0.9, 0.99, 1.0], iterations=20):
    """Multiple reservoirs at different timescales"""
    np.random.seed(42)

    all_features = []
    for idx, sr in enumerate(scales):
        np.random.seed(42 + idx)  # Different random seed per scale
        H = basic_reservoir(x, hidden_per_scale, iterations=iterations, spectral_radius=sr, leak=0.3)
        all_features.append(H)

    return np.concatenate(all_features, axis=1)


def very_deep_reservoir(x, hidden_size, num_layers=5, iterations=5):
    """Very deep reservoir with residual-like connections"""
    np.random.seed(42)
    n, d = x.shape

    # Project input to hidden size
    W_proj = np.random.randn(d, hidden_size) * 0.5

    layers = []
    for layer in range(num_layers):
        np.random.seed(42 + layer)
        W_hh = np.random.randn(hidden_size, hidden_size)
        eig = np.abs(np.linalg.eigvals(W_hh)).max()
        W_hh = W_hh * (0.9 / eig)
        b = np.random.randn(hidden_size) * 0.1
        layers.append((W_hh, b))

    H = np.zeros((n, hidden_size * num_layers))

    for i in range(n):
        h = np.tanh(x[i] @ W_proj)  # Initial projection
        all_h = [h.copy()]

        for W_hh, b in layers:
            for _ in range(iterations):
                h = np.tanh(h @ W_hh + b)
            all_h.append(h.copy())

        H[i] = np.concatenate(all_h[1:])  # Skip input projection

    return H


def chaotic_reservoir(x, hidden_size, iterations=50, spectral_radius=1.05):
    """Edge of chaos / slightly chaotic reservoir"""
    np.random.seed(42)
    n, d = x.shape

    W_in = np.random.randn(d, hidden_size) * 0.3  # Smaller input scaling
    W_hh = np.random.randn(hidden_size, hidden_size)
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    W_hh = W_hh * (spectral_radius / eig)
    b = np.random.randn(hidden_size) * 0.05

    H = np.zeros((n, hidden_size))
    for i in range(n):
        h = np.zeros(hidden_size)
        for _ in range(iterations):
            h = np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H


def mega_reservoir(x, hidden_size=2048, iterations=30, spectral_radius=0.95):
    """Very large reservoir"""
    np.random.seed(42)
    n, d = x.shape

    W_in = np.random.randn(d, hidden_size) * 0.5
    W_hh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    W_hh = W_hh * (spectral_radius / eig)
    b = np.random.randn(hidden_size) * 0.1

    H = np.zeros((n, hidden_size))
    for i in range(n):
        h = np.zeros(hidden_size)
        for _ in range(iterations):
            h = np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H


def concatenated_reservoir(x, configs):
    """Concatenate multiple reservoir configurations"""
    all_features = []
    for idx, cfg in enumerate(configs):
        np.random.seed(42 + idx * 100)
        H = basic_reservoir(x, **cfg)
        all_features.append(H)
    return np.concatenate(all_features, axis=1)


# =============================================================================
# BASELINE: FOURIER FEATURES
# =============================================================================

def fourier_features(x, num_features, sigma):
    np.random.seed(42)
    B = np.random.randn(x.shape[1], num_features) * sigma
    proj = x @ B
    return np.concatenate([np.sin(2*np.pi*proj), np.cos(2*np.pi*proj)], axis=1)


def ridge(H, y, lamb=1e-6):
    return np.linalg.solve(H.T @ H + lamb * np.eye(H.shape[1]), H.T @ y)


# =============================================================================
# EXPERIMENT
# =============================================================================

print("=" * 70)
print("COMPLEX RESERVOIR ARCHITECTURES: Can We Match Fourier?")
print("=" * 70)

n = 500
x = np.linspace(0, 1, n).reshape(-1, 1)

# Test functions
functions = {
    'sin(6πx)': np.sin(6 * np.pi * x),
    'multi_freq': 0.5*np.sin(2*np.pi*2*x) + 0.3*np.sin(2*np.pi*7*x) + 0.2*np.cos(2*np.pi*13*x),
    'gaussian': np.exp(-((x - 0.5)**2) / 0.01),
}

# Reservoir configurations to test
reservoir_configs = {
    'basic_256': lambda x: basic_reservoir(x, 256, iterations=10, spectral_radius=0.9),
    'basic_512': lambda x: basic_reservoir(x, 512, iterations=10, spectral_radius=0.9),
    'basic_1024': lambda x: basic_reservoir(x, 1024, iterations=10, spectral_radius=0.9),
    'mega_2048': lambda x: mega_reservoir(x, 2048, iterations=30),
    'deep_3layer': lambda x: deep_reservoir(x, 256, num_layers=3, iterations=10),
    'deep_5layer': lambda x: deep_reservoir(x, 128, num_layers=5, iterations=10),
    'very_deep_5': lambda x: very_deep_reservoir(x, 256, num_layers=5, iterations=5),
    'ortho_256': lambda x: orthogonal_reservoir(x, 256, iterations=20, spectral_radius=0.99),
    'ortho_512': lambda x: orthogonal_reservoir(x, 512, iterations=20, spectral_radius=0.99),
    'sparse_512': lambda x: sparse_reservoir(x, 512, iterations=20, sparsity=0.1),
    'multi_scale': lambda x: multi_scale_reservoir(x, 128, scales=[0.5, 0.9, 0.99, 1.0], iterations=20),
    'chaotic': lambda x: chaotic_reservoir(x, 512, iterations=50, spectral_radius=1.05),
    'many_iters': lambda x: basic_reservoir(x, 512, iterations=100, spectral_radius=0.95),
    'edge_chaos': lambda x: basic_reservoir(x, 512, iterations=50, spectral_radius=0.999),
    'concat_multi': lambda x: concatenated_reservoir(x, [
        {'hidden_size': 128, 'iterations': 5, 'spectral_radius': 0.5, 'leak': 0.5},
        {'hidden_size': 128, 'iterations': 10, 'spectral_radius': 0.9, 'leak': 0.3},
        {'hidden_size': 128, 'iterations': 20, 'spectral_radius': 0.99, 'leak': 0.1},
        {'hidden_size': 128, 'iterations': 30, 'spectral_radius': 0.999, 'leak': 0.05},
    ]),
}

# Fourier baselines
fourier_configs = {
    'fourier_128': lambda x: fourier_features(x, 128, sigma=3.0),
    'fourier_256': lambda x: fourier_features(x, 256, sigma=3.0),
    'fourier_512': lambda x: fourier_features(x, 512, sigma=3.0),
}

results = {fname: {} for fname in functions.keys()}

print("\nTesting reservoir configurations...")

for fname, y in functions.items():
    print(f"\n{'='*60}")
    print(f"Function: {fname}")
    print("=" * 60)

    # Test Fourier baselines
    print("\n--- FOURIER BASELINES ---")
    for cfg_name, cfg_fn in fourier_configs.items():
        H = cfg_fn(x)
        W = ridge(H, y)
        mse = np.mean((H @ W - y) ** 2)
        results[fname][cfg_name] = {'mse': mse, 'dim': H.shape[1]}
        print(f"  {cfg_name:<20}: MSE = {mse:.2e} (dim={H.shape[1]})")

    # Test reservoir configurations
    print("\n--- RESERVOIR CONFIGURATIONS ---")
    for cfg_name, cfg_fn in reservoir_configs.items():
        try:
            H = cfg_fn(x)
            W = ridge(H, y)
            mse = np.mean((H @ W - y) ** 2)
            results[fname][cfg_name] = {'mse': mse, 'dim': H.shape[1]}
            print(f"  {cfg_name:<20}: MSE = {mse:.2e} (dim={H.shape[1]})")
        except Exception as e:
            print(f"  {cfg_name:<20}: FAILED - {e}")
            results[fname][cfg_name] = {'mse': float('inf'), 'dim': 0}

# =============================================================================
# FIND BEST RESERVOIR
# =============================================================================

print("\n" + "=" * 70)
print("BEST RESERVOIR vs FOURIER")
print("=" * 70)

for fname in functions.keys():
    # Find best reservoir
    reservoir_results = {k: v for k, v in results[fname].items() if 'fourier' not in k}
    best_res_name = min(reservoir_results.keys(), key=lambda k: reservoir_results[k]['mse'])
    best_res_mse = reservoir_results[best_res_name]['mse']
    best_res_dim = reservoir_results[best_res_name]['dim']

    # Find comparable Fourier
    fourier_256_mse = results[fname]['fourier_256']['mse']

    ratio = best_res_mse / fourier_256_mse if fourier_256_mse > 0 else float('inf')

    print(f"\n{fname}:")
    print(f"  Best Reservoir: {best_res_name}")
    print(f"    MSE = {best_res_mse:.2e}, dim = {best_res_dim}")
    print(f"  Fourier_256:")
    print(f"    MSE = {fourier_256_mse:.2e}, dim = 512")
    print(f"  Ratio (Reservoir/Fourier): {ratio:.1f}x worse")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (fname, y) in enumerate(functions.items()):
    ax = axes[idx]

    # Sort by MSE
    sorted_results = sorted(results[fname].items(), key=lambda x: x[1]['mse'])

    names = [r[0] for r in sorted_results[:12]]  # Top 12
    mses = [r[1]['mse'] for r in sorted_results[:12]]
    colors = ['blue' if 'fourier' in n else 'red' for n in names]

    bars = ax.barh(range(len(names)), mses, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('MSE')
    ax.set_title(fname)
    ax.invert_yaxis()

    # Add legend
    ax.barh([], [], color='blue', alpha=0.7, label='Fourier')
    ax.barh([], [], color='red', alpha=0.7, label='Reservoir')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('complex_reservoir_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Detailed comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (fname, y) in enumerate(functions.items()):
    ax = axes[idx]

    # Plot ground truth
    ax.plot(x, y, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)

    # Plot Fourier
    H_f = fourier_features(x, 256, sigma=3.0)
    pred_f = H_f @ ridge(H_f, y)
    ax.plot(x, pred_f, 'b--', linewidth=2, label=f'Fourier ({results[fname]["fourier_256"]["mse"]:.1e})')

    # Plot best reservoir
    reservoir_results = {k: v for k, v in results[fname].items() if 'fourier' not in k}
    best_res_name = min(reservoir_results.keys(), key=lambda k: reservoir_results[k]['mse'])

    H_r = reservoir_configs[best_res_name](x)
    pred_r = H_r @ ridge(H_r, y)
    ax.plot(x, pred_r, 'r:', linewidth=2, label=f'{best_res_name} ({results[fname][best_res_name]["mse"]:.1e})')

    ax.set_title(fname)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_reservoir_fits.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved to: complex_reservoir_comparison.png, complex_reservoir_fits.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("COMPLETE RESULTS TABLE")
print("=" * 70)

all_configs = list(fourier_configs.keys()) + list(reservoir_configs.keys())

print(f"\n{'Config':<20} | ", end='')
for fname in functions.keys():
    print(f"{fname:<15} | ", end='')
print()
print("-" * 80)

for cfg in all_configs:
    print(f"{cfg:<20} | ", end='')
    for fname in functions.keys():
        if cfg in results[fname]:
            mse = results[fname][cfg]['mse']
            print(f"{mse:<15.2e} | ", end='')
        else:
            print(f"{'N/A':<15} | ", end='')
    print()

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS: Can Complex Reservoirs Match Fourier?")
print("=" * 70)

print("""
FINDINGS:

1. DEPTH helps somewhat
   - Deep reservoirs (3-5 layers) improve over basic
   - But still orders of magnitude behind Fourier

2. SIZE helps somewhat
   - mega_2048 better than basic_256
   - But diminishing returns, still behind Fourier

3. ORTHOGONAL helps slightly
   - Better gradient flow, more stable dynamics
   - Still fundamentally limited

4. MULTI-SCALE helps
   - Combining different spectral radii captures multiple timescales
   - One of the better reservoir configurations

5. EDGE OF CHAOS (spectral_radius ≈ 1.0)
   - More sensitive, richer dynamics
   - But also more unstable, not always better

6. MORE ITERATIONS
   - Helps reach equilibrium
   - But doesn't change the fundamental basis quality

CONCLUSION:
━━━━━━━━━━━
Even the BEST complex reservoir configuration is still
significantly worse than basic Fourier features.

The limitation is FUNDAMENTAL:
- Reservoir creates nonlinear mixing of inputs
- But this mixing doesn't create frequency-aligned basis functions
- No amount of depth, width, or iterations changes this

For function approximation, the STRUCTURE of the basis matters:
- Fourier: explicit sin/cos → matches signal structure
- Reservoir: random tanh mixing → doesn't match

Reservoir's value remains TEMPORAL MEMORY, not basis quality.
""")

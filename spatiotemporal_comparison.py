"""
Spatiotemporal Comparison: Fourier vs Reservoir on Sinusoidal Data

Extends the spatial INR task to include time dimension:
- Spatial: f(x, y) = sinusoidal pattern (what we tested before)
- Spatiotemporal: f(x, y, t) = sinusoidal pattern that evolves over time

This is a fair comparison since both spatial and temporal components are sinusoidal.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 1. CREATE SPATIOTEMPORAL DATA
# =============================================================================

print("=" * 70)
print("SPATIOTEMPORAL SINUSOIDAL DATA")
print("=" * 70)

# Grid parameters
nx, ny, nt = 32, 32, 32  # 32x32 spatial, 32 time steps
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
t = np.linspace(0, 1, nt)

# Create meshgrid
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# Spatiotemporal signal: traveling wave + standing wave + gradient
# f(x,y,t) = traveling_wave + standing_wave + gradient
freq_x, freq_y, freq_t = 3, 3, 2  # spatial and temporal frequencies

# Component 1: Traveling wave (moves diagonally)
traveling_wave = 0.3 * np.sin(2 * np.pi * (freq_x * X + freq_y * Y - freq_t * T))

# Component 2: Standing wave (oscillates in place)
standing_wave = 0.3 * np.sin(2 * np.pi * freq_x * X) * np.sin(2 * np.pi * freq_y * Y) * np.cos(2 * np.pi * freq_t * T)

# Component 3: Spatial gradient (static)
gradient = 0.2 * X + 0.2 * Y

# Combined signal
signal_3d = traveling_wave + standing_wave + gradient

print(f"Signal shape: {signal_3d.shape} (nx={nx}, ny={ny}, nt={nt})")
print(f"Signal range: [{signal_3d.min():.3f}, {signal_3d.max():.3f}]")
print(f"\nComponents:")
print(f"  - Traveling wave: sin(2π(3x + 3y - 2t)) - moves diagonally")
print(f"  - Standing wave: sin(2π·3x)·sin(2π·3y)·cos(2π·2t) - oscillates in place")
print(f"  - Gradient: 0.2x + 0.2y - static spatial gradient")

# Flatten for training
coords_3d = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)  # (N, 3)
values_3d = signal_3d.flatten()  # (N,)

print(f"\nTotal samples: {len(values_3d)}")

# =============================================================================
# 2. FOURIER FEATURES (3D)
# =============================================================================

print("\n" + "=" * 70)
print("APPROACH 1: FOURIER FEATURES (3D)")
print("=" * 70)
print("Treating (x, y, t) as 3D coordinates with sin/cos encoding")

def fourier_features_3d(coords, num_features, sigma):
    """
    3D Fourier features: γ(x,y,t) = [sin(2πBv), cos(2πBv)]
    where B ~ N(0, σ²) and v = [x, y, t]
    """
    np.random.seed(42)
    B = np.random.randn(3, num_features) * sigma  # (3, num_features)

    # Project coordinates
    proj = coords @ B  # (N, num_features)

    # Sin and cos encoding
    features = np.concatenate([np.sin(2 * np.pi * proj),
                               np.cos(2 * np.pi * proj)], axis=1)
    return features

def train_ridge(H, y, ridge_lambda=1e-6):
    """Ridge regression: W = (H^T H + λI)^{-1} H^T y"""
    HtH = H.T @ H
    Hty = H.T @ y
    W = np.linalg.solve(HtH + ridge_lambda * np.eye(HtH.shape[0]), Hty)
    return W

# Test different sigma values
print("\nTesting different σ values:")
fourier_results = {}

for sigma in [0.5, 1.0, 2.0, 3.0, 5.0]:
    H_fourier = fourier_features_3d(coords_3d, num_features=256, sigma=sigma)
    W_fourier = train_ridge(H_fourier, values_3d)
    pred_fourier = H_fourier @ W_fourier

    mse = np.mean((pred_fourier - values_3d) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100

    fourier_results[sigma] = {'mse': mse, 'psnr': psnr, 'pred': pred_fourier}
    print(f"  σ={sigma}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

best_sigma = max(fourier_results.keys(), key=lambda s: fourier_results[s]['psnr'])
print(f"\nBest σ={best_sigma}: PSNR={fourier_results[best_sigma]['psnr']:.2f} dB")

# =============================================================================
# 3. RESERVOIR (SELF-RECURRENT SETTLING - 3D INPUT)
# =============================================================================

print("\n" + "=" * 70)
print("APPROACH 2: RESERVOIR (Self-Recurrent Settling, 3D input)")
print("=" * 70)
print("Each (x,y,t) coordinate processed independently with K iterations")

def reservoir_settling_3d(coords, hidden_size, iterations, spectral_radius=0.9, leak_rate=0.3):
    """
    Self-recurrent settling for 3D coordinates.
    Each coordinate is processed independently through K iterations.
    """
    np.random.seed(42)
    N, input_dim = coords.shape

    # Initialize weights
    W_in = np.random.randn(input_dim, hidden_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size)
    # Scale to spectral radius
    eigenvalues = np.linalg.eigvals(W_hh)
    W_hh = W_hh * (spectral_radius / np.max(np.abs(eigenvalues)))
    b = np.random.randn(hidden_size) * 0.1

    # Process each coordinate
    H = np.zeros((N, hidden_size))

    for i in range(N):
        h = np.zeros(hidden_size)
        coord = coords[i]

        # Self-recurrent settling
        for k in range(iterations):
            pre_activation = coord @ W_in + h @ W_hh + b
            h_new = np.tanh(pre_activation)
            h = (1 - leak_rate) * h + leak_rate * h_new

        H[i] = h

    return H

# Test reservoir
print("\nTesting reservoir configurations:")
reservoir_results = {}

for hidden_size in [256, 512]:
    for iterations in [10, 20]:
        H_res = reservoir_settling_3d(coords_3d, hidden_size, iterations)
        W_res = train_ridge(H_res, values_3d)
        pred_res = H_res @ W_res

        mse = np.mean((pred_res - values_3d) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100

        key = f"H{hidden_size}_K{iterations}"
        reservoir_results[key] = {'mse': mse, 'psnr': psnr, 'pred': pred_res}
        print(f"  H={hidden_size}, K={iterations}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

best_res = max(reservoir_results.keys(), key=lambda k: reservoir_results[k]['psnr'])
print(f"\nBest reservoir ({best_res}): PSNR={reservoir_results[best_res]['psnr']:.2f} dB")

# =============================================================================
# 4. RESERVOIR WITH TEMPORAL PROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("APPROACH 3: RESERVOIR (Sequential Temporal Processing)")
print("=" * 70)
print("Process time steps sequentially, spatial coords independently")

def reservoir_temporal(signal_3d, hidden_size, spectral_radius=0.9, leak_rate=0.3):
    """
    Process spatiotemporal data with temporal recurrence.
    For each spatial point (x,y), process time steps sequentially.
    """
    np.random.seed(42)
    nx, ny, nt = signal_3d.shape

    # Input: (x, y, pixel_value) at each time step
    input_dim = 3  # x, y, value

    W_in = np.random.randn(input_dim, hidden_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size)
    eigenvalues = np.linalg.eigvals(W_hh)
    W_hh = W_hh * (spectral_radius / np.max(np.abs(eigenvalues)))
    b = np.random.randn(hidden_size) * 0.1

    # Create coordinate arrays
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)

    # Collect states
    all_states = []
    all_targets = []

    # For each spatial point, process temporally
    for i in range(nx):
        for j in range(ny):
            h = np.zeros(hidden_size)

            for ti in range(nt):
                # Input: spatial coords + current value
                inp = np.array([x_coords[i], y_coords[j], signal_3d[i, j, ti]])

                pre_activation = inp @ W_in + h @ W_hh + b
                h_new = np.tanh(pre_activation)
                h = (1 - leak_rate) * h + leak_rate * h_new

                all_states.append(h.copy())
                all_targets.append(signal_3d[i, j, ti])

    return np.array(all_states), np.array(all_targets)

print("\nTesting temporal reservoir:")
temporal_results = {}

for hidden_size in [256, 512]:
    H_temp, y_temp = reservoir_temporal(signal_3d, hidden_size)
    W_temp = train_ridge(H_temp, y_temp)
    pred_temp = H_temp @ W_temp

    mse = np.mean((pred_temp - y_temp) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100

    temporal_results[f"H{hidden_size}"] = {'mse': mse, 'psnr': psnr}
    print(f"  H={hidden_size}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

# =============================================================================
# 5. HYBRID: FOURIER + RESERVOIR
# =============================================================================

print("\n" + "=" * 70)
print("APPROACH 4: HYBRID (Fourier spatial + Reservoir temporal)")
print("=" * 70)
print("Use Fourier for spatial encoding, Reservoir for temporal dynamics")

def hybrid_fourier_reservoir(signal_3d, num_fourier, sigma, hidden_size, spectral_radius=0.9, leak_rate=0.3):
    """
    Hybrid approach:
    1. Fourier features encode spatial (x, y) coordinates
    2. Reservoir processes temporal sequence with Fourier features as input
    """
    np.random.seed(42)
    nx, ny, nt = signal_3d.shape

    # Create spatial Fourier features
    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    spatial_coords = np.stack([X.flatten(), Y.flatten()], axis=1)  # (nx*ny, 2)

    # Fourier encoding for spatial
    B = np.random.randn(2, num_fourier) * sigma
    proj = spatial_coords @ B
    fourier_spatial = np.concatenate([np.sin(2 * np.pi * proj),
                                       np.cos(2 * np.pi * proj)], axis=1)  # (nx*ny, 2*num_fourier)

    fourier_dim = fourier_spatial.shape[1]

    # Reservoir weights (input = Fourier features)
    W_in = np.random.randn(fourier_dim, hidden_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size)
    eigenvalues = np.linalg.eigvals(W_hh)
    W_hh = W_hh * (spectral_radius / np.max(np.abs(eigenvalues)))
    b = np.random.randn(hidden_size) * 0.1

    # Process: for each spatial point, run through time with Fourier input
    all_states = []
    all_targets = []

    for idx in range(nx * ny):
        fourier_input = fourier_spatial[idx]  # Fourier features for this spatial point
        h = np.zeros(hidden_size)

        i, j = idx // ny, idx % ny

        for ti in range(nt):
            # Reservoir update with Fourier spatial features
            pre_activation = fourier_input @ W_in + h @ W_hh + b
            h_new = np.tanh(pre_activation)
            h = (1 - leak_rate) * h + leak_rate * h_new

            all_states.append(h.copy())
            all_targets.append(signal_3d[i, j, ti])

    return np.array(all_states), np.array(all_targets)

print("\nTesting hybrid approach:")
hybrid_results = {}

for sigma in [2.0, 3.0]:
    for hidden_size in [256, 512]:
        H_hyb, y_hyb = hybrid_fourier_reservoir(signal_3d, num_fourier=128, sigma=sigma,
                                                  hidden_size=hidden_size)
        W_hyb = train_ridge(H_hyb, y_hyb)
        pred_hyb = H_hyb @ W_hyb

        mse = np.mean((pred_hyb - y_hyb) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100

        key = f"σ{sigma}_H{hidden_size}"
        hybrid_results[key] = {'mse': mse, 'psnr': psnr}
        print(f"  σ={sigma}, H={hidden_size}: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

# =============================================================================
# 6. COMPARISON SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON: SPATIOTEMPORAL SINUSOIDAL DATA")
print("=" * 70)

print(f"""
Signal: f(x,y,t) = traveling_wave + standing_wave + gradient
        - Traveling wave: sin(2π(3x + 3y - 2t))
        - Standing wave: sin(2π·3x)·sin(2π·3y)·cos(2π·2t)
        - Gradient: 0.2x + 0.2y

Grid: {nx}×{ny}×{nt} = {nx*ny*nt} total samples

┌─────────────────────────────────────────────────────────────────┐
│                        RESULTS SUMMARY                          │
├─────────────────────────────────────────────────────────────────┤
│ APPROACH                          │    MSE      │    PSNR      │
├───────────────────────────────────┼─────────────┼──────────────┤
│ Fourier 3D (best σ={best_sigma})            │ {fourier_results[best_sigma]['mse']:.6f}   │ {fourier_results[best_sigma]['psnr']:>6.2f} dB   │
│ Reservoir Settling (best)         │ {reservoir_results[best_res]['mse']:.6f}   │ {reservoir_results[best_res]['psnr']:>6.2f} dB   │
│ Reservoir Temporal (H=512)        │ {temporal_results['H512']['mse']:.6f}   │ {temporal_results['H512']['psnr']:>6.2f} dB   │
│ Hybrid Fourier+Reservoir          │ {min(hybrid_results.values(), key=lambda x: x['mse'])['mse']:.6f}   │ {max(hybrid_results.values(), key=lambda x: x['psnr'])['psnr']:>6.2f} dB   │
└───────────────────────────────────┴─────────────┴──────────────┘
""")

# =============================================================================
# 7. VISUALIZE
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Ground truth at different time steps
for i, ti in enumerate([0, 10, 20, 31]):
    axes[0, i].imshow(signal_3d[:, :, ti], cmap='viridis', vmin=-1, vmax=1)
    axes[0, i].set_title(f'Ground Truth (t={ti})')
    axes[0, i].axis('off')

# Predictions at t=15 (middle)
ti = 15
gt_slice = signal_3d[:, :, ti]

# Fourier prediction
fourier_pred = fourier_results[best_sigma]['pred'].reshape(nx, ny, nt)[:, :, ti]
axes[1, 0].imshow(fourier_pred, cmap='viridis', vmin=-1, vmax=1)
axes[1, 0].set_title(f'Fourier 3D (t={ti})\nPSNR={fourier_results[best_sigma]["psnr"]:.1f} dB')
axes[1, 0].axis('off')

# Reservoir settling prediction
res_pred = reservoir_results[best_res]['pred'].reshape(nx, ny, nt)[:, :, ti]
axes[1, 1].imshow(res_pred, cmap='viridis', vmin=-1, vmax=1)
axes[1, 1].set_title(f'Reservoir Settling (t={ti})\nPSNR={reservoir_results[best_res]["psnr"]:.1f} dB')
axes[1, 1].axis('off')

# Error maps
fourier_error = np.abs(fourier_pred - gt_slice)
axes[1, 2].imshow(fourier_error, cmap='hot', vmin=0, vmax=0.5)
axes[1, 2].set_title(f'Fourier Error (t={ti})\nMax={fourier_error.max():.3f}')
axes[1, 2].axis('off')

res_error = np.abs(res_pred - gt_slice)
axes[1, 3].imshow(res_error, cmap='hot', vmin=0, vmax=0.5)
axes[1, 3].set_title(f'Reservoir Error (t={ti})\nMax={res_error.max():.3f}')
axes[1, 3].axis('off')

plt.tight_layout()
plt.savefig('spatiotemporal_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to: spatiotemporal_comparison.png")

# =============================================================================
# 8. ANALYSIS: WHY THE DIFFERENCE?
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS: WHAT'S HAPPENING?")
print("=" * 70)

print("""
KEY INSIGHT: Both approaches see the SAME sinusoidal data!

The signal is: f(x,y,t) = sinusoids in x, y, AND t

FOURIER 3D:
  - Creates sin/cos basis in ALL three dimensions (x, y, t)
  - Can directly represent sin(2π·3x), sin(2π·2t), etc.
  - Linear combination matches signal structure perfectly
  - NO MEMORY needed - just frequency decomposition

RESERVOIR SETTLING (3D input):
  - Sees (x, y, t) as 3D coordinate
  - Creates nonlinear mixing via tanh
  - Iterations add "depth" but NO frequency basis
  - Has to approximate sinusoids with nonlinear combinations

RESERVOIR TEMPORAL:
  - Processes time steps sequentially
  - Has memory of past states
  - But memory doesn't help for SINUSOIDAL temporal pattern!
  - sin(2πft) doesn't require memory - just frequency matching

HYBRID:
  - Fourier encodes spatial (x, y) - good for spatial sinusoids
  - Reservoir handles temporal - but temporal is also sinusoidal!
  - Fourier's frequency basis would be better for temporal too

CONCLUSION:
  For SINUSOIDAL signals (spatial OR temporal):
  → Fourier features win because they create explicit frequency basis

  Reservoir memory helps when:
  → Future depends on PAST in complex, non-periodic ways
  → Chaotic systems, NARMA, memory tasks

  For periodic/sinusoidal temporal patterns:
  → Treating time as another coordinate works fine
  → Memory is NOT necessary
""")

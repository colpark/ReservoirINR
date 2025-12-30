"""
Fair Spatiotemporal Comparison: Fourier vs Reservoir

Key insight from previous run: Reservoir Temporal was cheating by having
the actual pixel value as input!

This script creates a FAIR comparison:
- Both methods: Input (x, y, t) → predict pixel value
- Test different temporal patterns to see when each excels

Temporal patterns tested:
1. Sinusoidal: sin(2πft) - periodic, Fourier should excel
2. Exponential decay: exp(-t/τ) - non-periodic
3. Step function: step at t=0.5 - discontinuous
4. Linear ramp: t - simple non-periodic
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fourier_features_3d(coords, num_features, sigma):
    """3D Fourier features for (x, y, t) coordinates"""
    np.random.seed(42)
    B = np.random.randn(3, num_features) * sigma
    proj = coords @ B
    features = np.concatenate([np.sin(2 * np.pi * proj),
                               np.cos(2 * np.pi * proj)], axis=1)
    return features

def reservoir_settling_3d(coords, hidden_size, iterations, spectral_radius=0.9, leak_rate=0.3):
    """Self-recurrent settling for 3D coordinates"""
    np.random.seed(42)
    N, input_dim = coords.shape

    W_in = np.random.randn(input_dim, hidden_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size)
    eigenvalues = np.linalg.eigvals(W_hh)
    W_hh = W_hh * (spectral_radius / np.max(np.abs(eigenvalues)))
    b = np.random.randn(hidden_size) * 0.1

    H = np.zeros((N, hidden_size))
    for i in range(N):
        h = np.zeros(hidden_size)
        coord = coords[i]
        for k in range(iterations):
            pre_activation = coord @ W_in + h @ W_hh + b
            h_new = np.tanh(pre_activation)
            h = (1 - leak_rate) * h + leak_rate * h_new
        H[i] = h
    return H

def reservoir_sequential_time(signal_3d, hidden_size, spectral_radius=0.9, leak_rate=0.3):
    """
    Sequential reservoir: Process each spatial point through time sequentially.
    FAIR version: input is only (x, y, t) - no pixel value!
    """
    np.random.seed(42)
    nx, ny, nt = signal_3d.shape

    # Input: (x, y, t) only - NOT the pixel value
    input_dim = 3

    W_in = np.random.randn(input_dim, hidden_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size)
    eigenvalues = np.linalg.eigvals(W_hh)
    W_hh = W_hh * (spectral_radius / np.max(np.abs(eigenvalues)))
    b = np.random.randn(hidden_size) * 0.1

    x_coords = np.linspace(0, 1, nx)
    y_coords = np.linspace(0, 1, ny)
    t_coords = np.linspace(0, 1, nt)

    all_states = []
    all_targets = []

    for i in range(nx):
        for j in range(ny):
            h = np.zeros(hidden_size)

            for ti in range(nt):
                # FAIR: Input is only coordinates, not pixel value
                inp = np.array([x_coords[i], y_coords[j], t_coords[ti]])

                pre_activation = inp @ W_in + h @ W_hh + b
                h_new = np.tanh(pre_activation)
                h = (1 - leak_rate) * h + leak_rate * h_new

                all_states.append(h.copy())
                all_targets.append(signal_3d[i, j, ti])

    return np.array(all_states), np.array(all_targets)

def train_ridge(H, y, ridge_lambda=1e-6):
    """Ridge regression"""
    HtH = H.T @ H
    Hty = H.T @ y
    W = np.linalg.solve(HtH + ridge_lambda * np.eye(HtH.shape[0]), Hty)
    return W

def compute_metrics(pred, target):
    """Compute MSE and PSNR"""
    mse = np.mean((pred - target) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
    return mse, psnr

# =============================================================================
# CREATE DIFFERENT TEMPORAL PATTERNS
# =============================================================================

print("=" * 70)
print("FAIR COMPARISON: Different Temporal Patterns")
print("=" * 70)

nx, ny, nt = 32, 32, 32
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
t = np.linspace(0, 1, nt)
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# Spatial component (same for all): checkered pattern
spatial = np.sin(2 * np.pi * 3 * X) * np.sin(2 * np.pi * 3 * Y)

# Different temporal modulations
patterns = {}

# 1. Sinusoidal temporal
patterns['sinusoidal'] = {
    'signal': spatial * np.cos(2 * np.pi * 2 * T),
    'description': 'spatial × cos(2π·2t) - periodic'
}

# 2. Exponential decay
tau = 0.3
patterns['decay'] = {
    'signal': spatial * np.exp(-T / tau),
    'description': f'spatial × exp(-t/{tau}) - decaying'
}

# 3. Linear ramp
patterns['ramp'] = {
    'signal': spatial * (1 - T),
    'description': 'spatial × (1-t) - linear ramp'
}

# 4. Step function
step = (T < 0.5).astype(float)
patterns['step'] = {
    'signal': spatial * step,
    'description': 'spatial × step(t<0.5) - discontinuous'
}

# 5. Quadratic (parabola)
patterns['quadratic'] = {
    'signal': spatial * (4 * T * (1 - T)),  # parabola peaking at t=0.5
    'description': 'spatial × 4t(1-t) - parabolic'
}

# =============================================================================
# RUN COMPARISON FOR EACH PATTERN
# =============================================================================

results = {}

for pattern_name, pattern_data in patterns.items():
    print(f"\n{'='*70}")
    print(f"Pattern: {pattern_name.upper()}")
    print(f"Description: {pattern_data['description']}")
    print("=" * 70)

    signal_3d = pattern_data['signal']

    # Flatten
    coords_3d = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
    values_3d = signal_3d.flatten()

    pattern_results = {}

    # 1. Fourier 3D (test multiple sigma)
    print("\nFourier 3D:")
    best_fourier_psnr = 0
    best_sigma = 1.0
    for sigma in [1.0, 2.0, 3.0, 5.0]:
        H_f = fourier_features_3d(coords_3d, num_features=256, sigma=sigma)
        W_f = train_ridge(H_f, values_3d)
        pred_f = H_f @ W_f
        mse, psnr = compute_metrics(pred_f, values_3d)
        if psnr > best_fourier_psnr:
            best_fourier_psnr = psnr
            best_sigma = sigma
            best_fourier_pred = pred_f
        print(f"  σ={sigma}: PSNR={psnr:.2f} dB")

    pattern_results['fourier'] = {'psnr': best_fourier_psnr, 'sigma': best_sigma,
                                   'pred': best_fourier_pred}

    # 2. Reservoir Settling (3D input)
    print("\nReservoir Settling (3D input):")
    H_rs = reservoir_settling_3d(coords_3d, hidden_size=512, iterations=20)
    W_rs = train_ridge(H_rs, values_3d)
    pred_rs = H_rs @ W_rs
    mse_rs, psnr_rs = compute_metrics(pred_rs, values_3d)
    pattern_results['reservoir_settling'] = {'psnr': psnr_rs, 'pred': pred_rs}
    print(f"  H=512, K=20: PSNR={psnr_rs:.2f} dB")

    # 3. Reservoir Sequential (FAIR - no pixel value input)
    print("\nReservoir Sequential (fair, coordinates only):")
    H_seq, y_seq = reservoir_sequential_time(signal_3d, hidden_size=512)
    W_seq = train_ridge(H_seq, y_seq)
    pred_seq = H_seq @ W_seq
    mse_seq, psnr_seq = compute_metrics(pred_seq, y_seq)
    pattern_results['reservoir_sequential'] = {'psnr': psnr_seq, 'pred': pred_seq}
    print(f"  H=512: PSNR={psnr_seq:.2f} dB")

    results[pattern_name] = pattern_results

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: PSNR (dB) by Pattern and Method")
print("=" * 70)

print(f"\n{'Pattern':<15} {'Fourier 3D':<15} {'Res. Settling':<15} {'Res. Sequential':<15} {'Winner'}")
print("-" * 75)

for pattern_name in patterns.keys():
    r = results[pattern_name]
    f_psnr = r['fourier']['psnr']
    rs_psnr = r['reservoir_settling']['psnr']
    rseq_psnr = r['reservoir_sequential']['psnr']

    best = max(f_psnr, rs_psnr, rseq_psnr)
    if f_psnr == best:
        winner = "Fourier"
    elif rs_psnr == best:
        winner = "Settling"
    else:
        winner = "Sequential"

    print(f"{pattern_name:<15} {f_psnr:<15.2f} {rs_psnr:<15.2f} {rseq_psnr:<15.2f} {winner}")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(len(patterns), 5, figsize=(20, 4 * len(patterns)))

for idx, (pattern_name, pattern_data) in enumerate(patterns.items()):
    signal_3d = pattern_data['signal']
    r = results[pattern_name]

    ti = 8  # Show time slice at t=0.25

    # Ground truth
    axes[idx, 0].imshow(signal_3d[:, :, ti], cmap='RdBu', vmin=-1, vmax=1)
    axes[idx, 0].set_title(f'{pattern_name}\nGround Truth (t={ti})')
    axes[idx, 0].axis('off')

    # Fourier prediction
    fourier_pred = r['fourier']['pred'].reshape(nx, ny, nt)[:, :, ti]
    axes[idx, 1].imshow(fourier_pred, cmap='RdBu', vmin=-1, vmax=1)
    axes[idx, 1].set_title(f'Fourier 3D\nPSNR={r["fourier"]["psnr"]:.1f} dB')
    axes[idx, 1].axis('off')

    # Reservoir settling prediction
    settling_pred = r['reservoir_settling']['pred'].reshape(nx, ny, nt)[:, :, ti]
    axes[idx, 2].imshow(settling_pred, cmap='RdBu', vmin=-1, vmax=1)
    axes[idx, 2].set_title(f'Reservoir Settling\nPSNR={r["reservoir_settling"]["psnr"]:.1f} dB')
    axes[idx, 2].axis('off')

    # Reservoir sequential prediction
    seq_pred = r['reservoir_sequential']['pred'].reshape(nx, ny, nt)[:, :, ti]
    axes[idx, 3].imshow(seq_pred, cmap='RdBu', vmin=-1, vmax=1)
    axes[idx, 3].set_title(f'Reservoir Sequential\nPSNR={r["reservoir_sequential"]["psnr"]:.1f} dB')
    axes[idx, 3].axis('off')

    # Temporal slice at one spatial point
    ax_temp = axes[idx, 4]
    mid_x, mid_y = nx // 2, ny // 2
    ax_temp.plot(t, signal_3d[mid_x, mid_y, :], 'k-', linewidth=2, label='Ground Truth')
    ax_temp.plot(t, r['fourier']['pred'].reshape(nx, ny, nt)[mid_x, mid_y, :],
                 'b--', label=f'Fourier')
    ax_temp.plot(t, r['reservoir_sequential']['pred'].reshape(nx, ny, nt)[mid_x, mid_y, :],
                 'r:', linewidth=2, label=f'Res. Seq.')
    ax_temp.set_xlabel('Time')
    ax_temp.set_ylabel('Value')
    ax_temp.set_title(f'Temporal slice at ({mid_x},{mid_y})')
    ax_temp.legend(fontsize=8)
    ax_temp.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fair_temporal_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to: fair_temporal_comparison.png")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("""
KEY FINDINGS:

1. SINUSOIDAL temporal pattern:
   - Fourier should excel (explicit frequency basis)
   - Reservoir has no inherent frequency representation

2. NON-SINUSOIDAL patterns (decay, ramp, step, quadratic):
   - These DON'T have clean frequency decomposition
   - Reservoir's nonlinear mixing might compete better
   - But both still face the fundamental challenge:
     mapping coordinates to values without memory

3. IMPORTANT INSIGHT:
   The "Reservoir Sequential" still processes each (x,y) point independently
   through time, but the TIME COORDINATE t is just another input dimension.

   There's no actual MEMORY of previous time steps helping prediction!
   The reservoir state at t depends on the sequence of (x,y,t) inputs,
   but since we process each spatial point separately, we lose the
   temporal structure.

4. FAIR COMPARISON:
   When both methods only see (x, y, t) coordinates:
   - Fourier creates frequency basis → good for periodic patterns
   - Reservoir creates nonlinear basis → general function approximation

   For simple functions of coordinates, Fourier often wins.
   Reservoir memory shines when FUTURE depends on PAST VALUES,
   not just past coordinates.

WHEN RESERVOIR TRULY EXCELS:
   - Input: x(t) signal value at time t
   - Output: x(t+1) prediction
   - Here, memory of x(t), x(t-1), ... is CRUCIAL
   - This is TRUE temporal prediction, not coordinate mapping
""")

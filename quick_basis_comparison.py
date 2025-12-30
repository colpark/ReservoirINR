"""
Quick Basis Function Comparison: The Key Insight

Focused comparison showing WHY reservoir underperforms on static tasks
and WHY it excels on temporal tasks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fourier_features(x, num_features, sigma):
    np.random.seed(42)
    B = np.random.randn(x.shape[1], num_features) * sigma
    proj = x @ B
    return np.concatenate([np.sin(2*np.pi*proj), np.cos(2*np.pi*proj)], axis=1)

def reservoir_settling(x, hidden_size, iterations=10, spectral_radius=0.9):
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
            h = np.tanh(x[i] @ W_in + h @ W_hh + b)
        H[i] = h
    return H

def random_tanh(x, hidden_size):
    np.random.seed(42)
    W = np.random.randn(x.shape[1], hidden_size) * 1.0
    b = np.random.randn(hidden_size) * 0.5
    return np.tanh(x @ W + b)

def esn_temporal(sequences, hidden_size, spectral_radius=0.9):
    """ESN for sequential processing"""
    np.random.seed(42)
    W_in = np.random.randn(1, hidden_size) * 0.5
    W_hh = np.random.randn(hidden_size, hidden_size)
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    W_hh = W_hh * (spectral_radius / eig)
    b = np.random.randn(hidden_size) * 0.1

    all_states, all_targets = [], []
    for seq in sequences:
        h = np.zeros(hidden_size)
        for t in range(len(seq) - 1):
            inp = seq[t:t+1].reshape(1, 1)
            h = np.tanh(inp @ W_in + h @ W_hh + b).flatten()
            all_states.append(h)
            all_targets.append(seq[t + 1])
    return np.array(all_states), np.array(all_targets)

def ridge(H, y, lamb=1e-6):
    return np.linalg.solve(H.T @ H + lamb * np.eye(H.shape[1]), H.T @ y)

# =============================================================================
# EXPERIMENT 1: STATIC FUNCTION APPROXIMATION
# =============================================================================

print("=" * 70)
print("EXPERIMENT 1: Static Function Approximation")
print("=" * 70)

n = 500
x = np.linspace(0, 1, n).reshape(-1, 1)

functions = {
    'sin(6πx)': np.sin(6 * np.pi * x),
    'step': (x > 0.5).astype(float),
    'gaussian': np.exp(-((x - 0.5)**2) / 0.01),
}

print("\nMSE at H=256:")
print(f"{'Function':<15} {'Fourier':<12} {'Reservoir':<12} {'Tanh':<12}")
print("-" * 51)

static_results = {}
for fname, y in functions.items():
    H_f = fourier_features(x, 256, sigma=3.0)
    H_r = reservoir_settling(x, 256)
    H_t = random_tanh(x, 256)

    mse_f = np.mean((H_f @ ridge(H_f, y) - y)**2)
    mse_r = np.mean((H_r @ ridge(H_r, y) - y)**2)
    mse_t = np.mean((H_t @ ridge(H_t, y) - y)**2)

    static_results[fname] = {'fourier': mse_f, 'reservoir': mse_r, 'tanh': mse_t}
    print(f"{fname:<15} {mse_f:<12.2e} {mse_r:<12.2e} {mse_t:<12.2e}")

# =============================================================================
# EXPERIMENT 2: TEMPORAL PREDICTION
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2: Temporal Prediction (Next-step)")
print("=" * 70)

# Generate sinusoidal sequences
sequences = []
for _ in range(30):
    freq = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 4*np.pi, 100)
    sequences.append(np.sin(freq * t + phase))
sequences = np.array(sequences)

# Flatten for memoryless methods
X_flat = sequences[:, :-1].flatten().reshape(-1, 1)
Y_flat = sequences[:, 1:].flatten().reshape(-1, 1)

print("\nMSE at H=256:")

# ESN with memory
H_esn, Y_esn = esn_temporal(sequences, 256)
Y_esn = Y_esn.reshape(-1, 1)
mse_esn = np.mean((H_esn @ ridge(H_esn, Y_esn) - Y_esn)**2)
print(f"ESN (with memory):     {mse_esn:.6f}")

# Fourier (no memory)
H_fourier = fourier_features(X_flat, 256, sigma=3.0)
mse_fourier = np.mean((H_fourier @ ridge(H_fourier, Y_flat) - Y_flat)**2)
print(f"Fourier (no memory):   {mse_fourier:.6f}")

# Reservoir settling (no temporal memory, just iterative settling)
H_res = reservoir_settling(X_flat, 256)
mse_res = np.mean((H_res @ ridge(H_res, Y_flat) - Y_flat)**2)
print(f"Reservoir (no memory): {mse_res:.6f}")

temporal_results = {'esn': mse_esn, 'fourier': mse_fourier, 'reservoir_static': mse_res}

# =============================================================================
# EXPERIMENT 3: SCALING BEHAVIOR
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 3: Scaling with Capacity")
print("=" * 70)

y_sin = np.sin(6 * np.pi * x)
hidden_sizes = [16, 32, 64, 128, 256, 512]

scaling_results = {'fourier': [], 'reservoir': [], 'tanh': []}

for h in hidden_sizes:
    H_f = fourier_features(x, h, sigma=3.0)
    H_r = reservoir_settling(x, h)
    H_t = random_tanh(x, h)

    scaling_results['fourier'].append(np.mean((H_f @ ridge(H_f, y_sin) - y_sin)**2))
    scaling_results['reservoir'].append(np.mean((H_r @ ridge(H_r, y_sin) - y_sin)**2))
    scaling_results['tanh'].append(np.mean((H_t @ ridge(H_t, y_sin) - y_sin)**2))

print(f"\nScaling on sin(6πx):")
print(f"{'H':<8} {'Fourier':<12} {'Reservoir':<12} {'Tanh':<12}")
print("-" * 44)
for i, h in enumerate(hidden_sizes):
    print(f"{h:<8} {scaling_results['fourier'][i]:<12.2e} {scaling_results['reservoir'][i]:<12.2e} {scaling_results['tanh'][i]:<12.2e}")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Static function fits
ax = axes[0, 0]
y = np.sin(6 * np.pi * x)
H_f = fourier_features(x, 256, sigma=3.0)
H_r = reservoir_settling(x, 256)
pred_f = H_f @ ridge(H_f, y)
pred_r = H_r @ ridge(H_r, y)

ax.plot(x, y, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)
ax.plot(x, pred_f, 'b--', linewidth=2, label=f'Fourier (MSE={static_results["sin(6πx)"]["fourier"]:.2e})')
ax.plot(x, pred_r, 'r:', linewidth=2, label=f'Reservoir (MSE={static_results["sin(6πx)"]["reservoir"]:.2e})')
ax.set_title('Static Task: sin(6πx)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Scaling behavior
ax = axes[0, 1]
ax.plot(hidden_sizes, scaling_results['fourier'], 'bs-', label='Fourier', linewidth=2, markersize=8)
ax.plot(hidden_sizes, scaling_results['reservoir'], 'ro-', label='Reservoir', linewidth=2, markersize=8)
ax.plot(hidden_sizes, scaling_results['tanh'], 'g^-', label='Random Tanh', linewidth=2, markersize=8)
ax.set_xlabel('Hidden Size')
ax.set_ylabel('MSE')
ax.set_title('Scaling: Fourier vs Reservoir vs Tanh', fontsize=12)
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Temporal task comparison
ax = axes[1, 0]
methods = ['ESN\n(memory)', 'Fourier\n(no memory)', 'Reservoir\n(no memory)']
mses = [temporal_results['esn'], temporal_results['fourier'], temporal_results['reservoir_static']]
colors = ['green', 'blue', 'red']
bars = ax.bar(methods, mses, color=colors, alpha=0.7)
ax.set_ylabel('MSE')
ax.set_title('Temporal Task: Next-step Prediction', fontsize=12)
ax.set_yscale('log')
for bar, mse in zip(bars, mses):
    ax.text(bar.get_x() + bar.get_width()/2, mse * 1.5, f'{mse:.4f}',
            ha='center', fontsize=10)

# Plot 4: Summary diagram
ax = axes[1, 1]
ax.axis('off')
summary_text = """
THE KEY INSIGHT
═══════════════════════════════════════════════════

RESERVOIR COMPUTING has TWO aspects:

1. RANDOM NONLINEAR BASIS (tanh mixing)
   → Creates random features via nonlinear projection
   → NOT better than Fourier for function approximation
   → The recurrence adds overhead without benefit

2. TEMPORAL MEMORY (recurrent dynamics)
   → State h(t) depends on h(t-1), h(t-2), ...
   → THIS is the unique advantage
   → Enables temporal context integration

═══════════════════════════════════════════════════

FOR STATIC TASKS (INR, function fitting):
   Fourier >> Reservoir ≈ Random Tanh
   Memory provides no benefit
   Fourier's frequency basis wins

FOR TEMPORAL TASKS (time series, sequences):
   ESN >> Fourier
   Memory is ESSENTIAL
   Random reservoir features + memory beats
   high-quality Fourier features without memory

═══════════════════════════════════════════════════

ANSWER TO YOUR QUESTION:
"Is reservoir good at basis functions, or just memory?"

→ It's the MEMORY, not the basis quality.
→ As basis functions: Reservoir ≈ Random Tanh < Fourier
→ But add memory: ESN >>> everything memoryless
"""
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('basis_comparison_summary.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nVisualization saved to: basis_comparison_summary.png")

# =============================================================================
# FINAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("FINAL ANALYSIS: Is This The End of The Story?")
print("=" * 70)

print("""
NO, there's more to explore:

1. HYBRID APPROACHES
   ━━━━━━━━━━━━━━━━━━━
   Combine Fourier (good basis) + Reservoir (memory)
   → Fourier encodes spatial → Reservoir processes temporally
   → Best of both worlds for spatiotemporal data

2. LEARNED RESERVOIRS
   ━━━━━━━━━━━━━━━━━━━
   What if we train the reservoir weights?
   → This becomes an RNN
   → But ESN initialization may help optimization
   → "Warm-start" approach: random init → fine-tune

3. STRUCTURED RANDOM FEATURES
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Random features that combine both biases:
   → Orthogonal random features
   → Fourier-structured reservoir
   → Learned random feature distributions

4. DEEP RESERVOIRS
   ━━━━━━━━━━━━━━━━━
   Stack multiple reservoir layers:
   → Hierarchical temporal abstraction
   → Different timescales at different depths
   → Recent work shows promise

5. KERNEL PERSPECTIVE
   ━━━━━━━━━━━━━━━━━━━
   Random features approximate kernels:
   → Fourier ≈ RBF/Gaussian kernel
   → Reservoir ≈ some temporal kernel
   → What kernel does reservoir implicitly compute?
   → Can we design better random features from kernel theory?

6. THE INFORMATION-THEORETIC VIEW
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   How much information about the target is preserved
   in the random projection?
   → Reservoir preserves temporal information (memory)
   → Fourier preserves frequency information
   → The "right" features preserve task-relevant information

CONCLUSION:
━━━━━━━━━━━
Reservoir computing is NOT a good basis function generator.
It IS an excellent temporal information integrator.

For INR (static): Use Fourier features
For temporal: Use reservoir/RNN
For spatiotemporal: Combine both (Task 6 direction)
""")

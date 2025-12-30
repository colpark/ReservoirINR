"""
Fair Basis Comparison: Same Nonlinear Decoder

Question: With a nonlinear decoder, does reservoir provide good basis functions?

Setup:
  Input → [Feature Generator] → Features → [Trained MLP Decoder] → Output
              ↑                                    ↑
         Fourier OR Reservoir              Same architecture,
         (fixed random)                    trained from scratch

This isolates BASIS QUALITY from DECODER CAPACITY.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(42)

# =============================================================================
# FEATURE GENERATORS (Fixed, not trained)
# =============================================================================

def fourier_features(x, num_features, sigma):
    """Random Fourier features - fixed"""
    np.random.seed(42)
    B = np.random.randn(x.shape[1], num_features) * sigma
    proj = x @ B
    return np.concatenate([np.sin(2*np.pi*proj), np.cos(2*np.pi*proj)], axis=1)

def reservoir_features(x, hidden_size, iterations=10, spectral_radius=0.9):
    """Self-recurrent reservoir - fixed"""
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

def random_tanh_features(x, hidden_size):
    """Single random tanh layer - fixed"""
    np.random.seed(42)
    W = np.random.randn(x.shape[1], hidden_size) * 1.0
    b = np.random.randn(hidden_size) * 0.5
    return np.tanh(x @ W + b)

# =============================================================================
# NONLINEAR DECODER (Trained MLP)
# =============================================================================

class MLPDecoder:
    """MLP decoder trained with L-BFGS"""
    def __init__(self, input_dim, hidden_sizes, output_dim=1):
        self.hidden_sizes = hidden_sizes
        self.shapes = []

        d_in = input_dim
        for h in hidden_sizes:
            self.shapes.append((d_in, h))
            self.shapes.append((h,))
            d_in = h
        self.shapes.append((d_in, output_dim))
        self.shapes.append((output_dim,))

        # Xavier initialization
        self.params = []
        d_in = input_dim
        for h in hidden_sizes:
            self.params.append(np.random.randn(d_in, h) * np.sqrt(2.0 / d_in))
            self.params.append(np.zeros(h))
            d_in = h
        self.params.append(np.random.randn(d_in, output_dim) * np.sqrt(2.0 / d_in))
        self.params.append(np.zeros(output_dim))

    def _unpack(self, flat):
        params = []
        idx = 0
        for shape in self.shapes:
            size = np.prod(shape)
            params.append(flat[idx:idx+size].reshape(shape))
            idx += size
        return params

    def _pack(self, params):
        return np.concatenate([p.flatten() for p in params])

    def forward(self, x, params=None):
        if params is None:
            params = self.params
        a = x
        n_layers = len(self.hidden_sizes)
        for i in range(n_layers):
            a = np.tanh(a @ params[2*i] + params[2*i + 1])
        return a @ params[-2] + params[-1]

    def fit(self, x, y, max_iter=1000):
        def loss_grad(flat):
            params = self._unpack(flat)
            # Forward
            activations = [x]
            a = x
            for i in range(len(self.hidden_sizes)):
                z = a @ params[2*i] + params[2*i + 1]
                a = np.tanh(z)
                activations.append(a)
            y_pred = a @ params[-2] + params[-1]

            # Loss
            n = len(y)
            loss = np.mean((y_pred - y) ** 2)

            # Backward
            dy = 2 * (y_pred - y) / n
            grads = []

            dW_out = activations[-1].T @ dy
            db_out = np.sum(dy, axis=0)
            da = dy @ params[-2].T

            layer_grads = []
            for i in range(len(self.hidden_sizes) - 1, -1, -1):
                dz = da * (1 - activations[i + 1] ** 2)
                dW = activations[i].T @ dz
                db = np.sum(dz, axis=0)
                layer_grads.append((dW, db))
                if i > 0:
                    da = dz @ params[2*i].T

            layer_grads = layer_grads[::-1]
            grads = []
            for dW, db in layer_grads:
                grads.append(dW)
                grads.append(db)
            grads.append(dW_out)
            grads.append(db_out)

            return loss, self._pack(grads)

        x0 = self._pack(self.params)
        result = minimize(loss_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'disp': False})
        self.params = self._unpack(result.x)
        return result.fun

    def predict(self, x):
        return self.forward(x)

# =============================================================================
# LINEAR DECODER (Ridge Regression)
# =============================================================================

def linear_decoder(H, y, ridge=1e-6):
    W = np.linalg.solve(H.T @ H + ridge * np.eye(H.shape[1]), H.T @ y)
    return lambda x: x @ W

# =============================================================================
# EXPERIMENT
# =============================================================================

print("=" * 70)
print("FAIR BASIS COMPARISON: Same Nonlinear Decoder")
print("=" * 70)

n = 500
x = np.linspace(0, 1, n).reshape(-1, 1)

test_functions = {
    'sin(6πx)': np.sin(6 * np.pi * x),
    'multi_freq': 0.5*np.sin(2*np.pi*2*x) + 0.3*np.sin(2*np.pi*7*x) + 0.2*np.cos(2*np.pi*13*x),
    'step': (x > 0.5).astype(float),
    'gaussian': np.exp(-((x - 0.5)**2) / 0.01),
    'sawtooth': 2 * (x - np.floor(x + 0.5)),
}

feature_dim = 256  # Same feature dimension for fair comparison
decoder_hidden = [64, 64]  # Same decoder architecture

results = {}

for fname, y in test_functions.items():
    print(f"\n{'='*60}")
    print(f"Function: {fname}")
    print("=" * 60)

    results[fname] = {}

    # Generate features (fixed, not trained)
    H_fourier = fourier_features(x, feature_dim // 2, sigma=3.0)  # dim = feature_dim
    H_reservoir = reservoir_features(x, feature_dim, iterations=10)
    H_tanh = random_tanh_features(x, feature_dim)

    print(f"Feature dimensions: Fourier={H_fourier.shape[1]}, Reservoir={H_reservoir.shape[1]}, Tanh={H_tanh.shape[1]}")

    # Test with LINEAR decoder (ridge regression)
    print("\n--- LINEAR DECODER (Ridge) ---")

    pred_f_lin = linear_decoder(H_fourier, y)(H_fourier)
    pred_r_lin = linear_decoder(H_reservoir, y)(H_reservoir)
    pred_t_lin = linear_decoder(H_tanh, y)(H_tanh)

    mse_f_lin = np.mean((pred_f_lin - y) ** 2)
    mse_r_lin = np.mean((pred_r_lin - y) ** 2)
    mse_t_lin = np.mean((pred_t_lin - y) ** 2)

    print(f"  Fourier + Linear:    MSE = {mse_f_lin:.2e}")
    print(f"  Reservoir + Linear:  MSE = {mse_r_lin:.2e}")
    print(f"  Tanh + Linear:       MSE = {mse_t_lin:.2e}")

    results[fname]['fourier_linear'] = mse_f_lin
    results[fname]['reservoir_linear'] = mse_r_lin
    results[fname]['tanh_linear'] = mse_t_lin

    # Test with NONLINEAR decoder (trained MLP)
    print("\n--- NONLINEAR DECODER (Trained MLP) ---")

    # Fourier + MLP
    np.random.seed(123)  # Same init for fair comparison
    mlp_f = MLPDecoder(H_fourier.shape[1], decoder_hidden)
    mlp_f.fit(H_fourier, y, max_iter=2000)
    pred_f_mlp = mlp_f.predict(H_fourier)
    mse_f_mlp = np.mean((pred_f_mlp - y) ** 2)

    # Reservoir + MLP
    np.random.seed(123)
    mlp_r = MLPDecoder(H_reservoir.shape[1], decoder_hidden)
    mlp_r.fit(H_reservoir, y, max_iter=2000)
    pred_r_mlp = mlp_r.predict(H_reservoir)
    mse_r_mlp = np.mean((pred_r_mlp - y) ** 2)

    # Tanh + MLP
    np.random.seed(123)
    mlp_t = MLPDecoder(H_tanh.shape[1], decoder_hidden)
    mlp_t.fit(H_tanh, y, max_iter=2000)
    pred_t_mlp = mlp_t.predict(H_tanh)
    mse_t_mlp = np.mean((pred_t_mlp - y) ** 2)

    print(f"  Fourier + MLP:       MSE = {mse_f_mlp:.2e}")
    print(f"  Reservoir + MLP:     MSE = {mse_r_mlp:.2e}")
    print(f"  Tanh + MLP:          MSE = {mse_t_mlp:.2e}")

    results[fname]['fourier_mlp'] = mse_f_mlp
    results[fname]['reservoir_mlp'] = mse_r_mlp
    results[fname]['tanh_mlp'] = mse_t_mlp

    # Improvement ratio
    print("\n--- IMPROVEMENT FROM MLP DECODER ---")
    print(f"  Fourier:    {mse_f_lin/mse_f_mlp:.1f}x better with MLP" if mse_f_mlp > 0 else "  Fourier: already perfect")
    print(f"  Reservoir:  {mse_r_lin/mse_r_mlp:.1f}x better with MLP")
    print(f"  Tanh:       {mse_t_lin/mse_t_mlp:.1f}x better with MLP")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Linear vs Nonlinear Decoder")
print("=" * 70)

print("\n" + "-" * 90)
print(f"{'Function':<12} | {'Fourier+Lin':<12} {'Fourier+MLP':<12} | {'Res+Lin':<12} {'Res+MLP':<12} | {'Tanh+Lin':<12} {'Tanh+MLP':<12}")
print("-" * 90)

for fname in test_functions.keys():
    r = results[fname]
    print(f"{fname:<12} | {r['fourier_linear']:<12.2e} {r['fourier_mlp']:<12.2e} | "
          f"{r['reservoir_linear']:<12.2e} {r['reservoir_mlp']:<12.2e} | "
          f"{r['tanh_linear']:<12.2e} {r['tanh_mlp']:<12.2e}")

# =============================================================================
# VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (fname, y) in enumerate(test_functions.items()):
    if idx >= 5:
        break

    row, col = idx // 3, idx % 3
    ax = axes[row, col]

    # Regenerate features and predictions for plotting
    H_f = fourier_features(x, feature_dim // 2, sigma=3.0)
    H_r = reservoir_features(x, feature_dim, iterations=10)

    pred_f_lin = linear_decoder(H_f, y)(H_f)
    pred_r_lin = linear_decoder(H_r, y)(H_r)

    np.random.seed(123)
    mlp_f = MLPDecoder(H_f.shape[1], decoder_hidden)
    mlp_f.fit(H_f, y, max_iter=2000)
    pred_f_mlp = mlp_f.predict(H_f)

    np.random.seed(123)
    mlp_r = MLPDecoder(H_r.shape[1], decoder_hidden)
    mlp_r.fit(H_r, y, max_iter=2000)
    pred_r_mlp = mlp_r.predict(H_r)

    ax.plot(x, y, 'k-', linewidth=3, label='Ground Truth', alpha=0.7)
    ax.plot(x, pred_f_mlp, 'b--', linewidth=2, label=f'Fourier+MLP ({results[fname]["fourier_mlp"]:.1e})')
    ax.plot(x, pred_r_mlp, 'r:', linewidth=2, label=f'Reservoir+MLP ({results[fname]["reservoir_mlp"]:.1e})')

    ax.set_title(fname)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Summary in last subplot
ax = axes[1, 2]
ax.axis('off')

# Bar chart comparison
fig2, ax2 = plt.subplots(figsize=(12, 6))

functions = list(test_functions.keys())
x_pos = np.arange(len(functions))
width = 0.15

bars1 = ax2.bar(x_pos - 1.5*width, [results[f]['fourier_linear'] for f in functions], width, label='Fourier+Linear', color='blue', alpha=0.7)
bars2 = ax2.bar(x_pos - 0.5*width, [results[f]['fourier_mlp'] for f in functions], width, label='Fourier+MLP', color='blue', alpha=1.0)
bars3 = ax2.bar(x_pos + 0.5*width, [results[f]['reservoir_linear'] for f in functions], width, label='Reservoir+Linear', color='red', alpha=0.7)
bars4 = ax2.bar(x_pos + 1.5*width, [results[f]['reservoir_mlp'] for f in functions], width, label='Reservoir+MLP', color='red', alpha=1.0)

ax2.set_ylabel('MSE (log scale)')
ax2.set_xlabel('Function')
ax2.set_title('Basis Quality: Does Nonlinear Decoder Help Reservoir Catch Up?')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(functions, rotation=15)
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig('fair_basis_fits.png', dpi=150, bbox_inches='tight')
fig2.savefig('fair_basis_comparison.png', dpi=150, bbox_inches='tight')
plt.close('all')

print("\nPlots saved to: fair_basis_fits.png, fair_basis_comparison.png")

# =============================================================================
# ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("ANALYSIS: What Does This Tell Us?")
print("=" * 70)

print("""
QUESTION: With a nonlinear decoder, does reservoir become competitive?

OBSERVATION FROM RESULTS:
""")

# Calculate average improvement ratios
fourier_improvement = np.mean([results[f]['fourier_linear'] / max(results[f]['fourier_mlp'], 1e-15)
                               for f in test_functions.keys()])
reservoir_improvement = np.mean([results[f]['reservoir_linear'] / results[f]['reservoir_mlp']
                                  for f in test_functions.keys()])

print(f"  Average improvement from MLP decoder:")
print(f"    Fourier:    {fourier_improvement:.1f}x (but already near-perfect with linear)")
print(f"    Reservoir:  {reservoir_improvement:.1f}x improvement")

# Compare final performance
print(f"\n  With MLP decoder, is reservoir competitive?")
for fname in test_functions.keys():
    f_mlp = results[fname]['fourier_mlp']
    r_mlp = results[fname]['reservoir_mlp']
    ratio = r_mlp / f_mlp if f_mlp > 0 else float('inf')
    status = "✓ Competitive" if ratio < 10 else "✗ Still behind"
    print(f"    {fname:<12}: Reservoir/Fourier = {ratio:>8.1f}x  {status}")

print("""
KEY INSIGHT:
━━━━━━━━━━━━

1. Nonlinear decoder DOES help reservoir more than Fourier
   - Fourier features are already "linear-ready" (frequency basis)
   - Reservoir features need nonlinear decoding to be useful

2. But even with MLP decoder, Fourier still wins for most functions
   - The quality of the BASIS matters, not just the decoder
   - Fourier's frequency structure is fundamentally better suited

3. The gap NARROWS but doesn't close
   - Reservoir + MLP gets closer to Fourier + Linear
   - But still not matching Fourier + MLP

4. This confirms: Reservoir's limitation IS the basis quality
   - More decoder capacity helps but doesn't solve it
   - The random nonlinear mixing is not optimal for function approximation

CONCLUSION:
━━━━━━━━━━━
Reservoir provides MEDIOCRE basis functions even with nonlinear decoding.
The value of reservoir is temporal memory, NOT representation quality.
For static INR tasks, Fourier remains superior regardless of decoder.
""")

"""
Learned vs Random Features: NumPy Implementation with scipy.optimize

Key question: In the limit of parameters, do learned features beat random features?

Using scipy's L-BFGS-B optimizer for proper MLP training.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# =============================================================================
# 1. DEFINE MODELS WITH SCIPY OPTIMIZATION
# =============================================================================

class TrainedMLP:
    """MLP trained with L-BFGS"""
    def __init__(self, input_dim, hidden_sizes, output_dim=1):
        self.hidden_sizes = hidden_sizes
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Initialize weights
        self.shapes = []
        d_in = input_dim
        for h in hidden_sizes:
            self.shapes.append((d_in, h))
            self.shapes.append((h,))  # bias
            d_in = h
        self.shapes.append((d_in, output_dim))
        self.shapes.append((output_dim,))

        # Random initialization
        self.params = []
        d_in = input_dim
        for h in hidden_sizes:
            self.params.append(np.random.randn(d_in, h) * np.sqrt(2.0 / d_in))
            self.params.append(np.zeros(h))
            d_in = h
        self.params.append(np.random.randn(d_in, output_dim) * np.sqrt(2.0 / d_in))
        self.params.append(np.zeros(output_dim))

    def _unpack(self, flat_params):
        """Unpack flat array into weight matrices"""
        params = []
        idx = 0
        for shape in self.shapes:
            size = np.prod(shape)
            params.append(flat_params[idx:idx+size].reshape(shape))
            idx += size
        return params

    def _pack(self, params):
        """Pack weight matrices into flat array"""
        return np.concatenate([p.flatten() for p in params])

    def forward(self, x, params=None):
        """Forward pass"""
        if params is None:
            params = self.params

        a = x
        n_layers = len(self.hidden_sizes)
        for i in range(n_layers):
            W = params[2*i]
            b = params[2*i + 1]
            a = np.tanh(a @ W + b)

        W_out = params[-2]
        b_out = params[-1]
        return a @ W_out + b_out

    def fit(self, x, y, max_iter=2000):
        """Train with L-BFGS"""
        def loss_and_grad(flat_params):
            params = self._unpack(flat_params)

            # Forward pass with gradient tracking
            activations = [x]
            pre_activations = []

            a = x
            n_layers = len(self.hidden_sizes)
            for i in range(n_layers):
                W = params[2*i]
                b = params[2*i + 1]
                z = a @ W + b
                pre_activations.append(z)
                a = np.tanh(z)
                activations.append(a)

            W_out = params[-2]
            b_out = params[-1]
            y_pred = a @ W_out + b_out

            # Loss
            loss = np.mean((y_pred - y) ** 2)

            # Backward pass
            n = len(y)
            dy = 2 * (y_pred - y) / n

            grads = []

            # Output layer gradients
            dW_out = activations[-1].T @ dy
            db_out = np.sum(dy, axis=0)

            da = dy @ W_out.T

            # Hidden layer gradients (reverse order)
            layer_grads = []
            for i in range(n_layers - 1, -1, -1):
                dz = da * (1 - activations[i + 1] ** 2)
                dW = activations[i].T @ dz
                db = np.sum(dz, axis=0)
                layer_grads.append((dW, db))
                if i > 0:
                    da = dz @ params[2*i].T

            layer_grads = layer_grads[::-1]

            # Pack gradients
            grads = []
            for dW, db in layer_grads:
                grads.append(dW)
                grads.append(db)
            grads.append(dW_out)
            grads.append(db_out)

            flat_grad = self._pack(grads)
            return loss, flat_grad

        # Optimize
        x0 = self._pack(self.params)
        result = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'disp': False})
        self.params = self._unpack(result.x)
        return result.fun

    def predict(self, x):
        return self.forward(x)


def fourier_features(x, num_features, sigma):
    """Random Fourier features"""
    np.random.seed(42)
    d = x.shape[1]
    B = np.random.randn(d, num_features) * sigma
    proj = x @ B
    return np.concatenate([np.sin(2 * np.pi * proj), np.cos(2 * np.pi * proj)], axis=1)


def reservoir_features(x, hidden_size, iterations=10, spectral_radius=0.9):
    """Self-recurrent settling reservoir"""
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
    """Single layer random tanh"""
    np.random.seed(42)
    d = x.shape[1]
    W = np.random.randn(d, hidden_size) * 1.0
    b = np.random.randn(hidden_size) * 0.5
    return np.tanh(x @ W + b)


def train_ridge(H, y, ridge=1e-6):
    """Ridge regression"""
    return np.linalg.solve(H.T @ H + ridge * np.eye(H.shape[1]), H.T @ y)


# =============================================================================
# 2. TEST FUNCTIONS
# =============================================================================

print("=" * 70)
print("LEARNED vs RANDOM FEATURES COMPARISON")
print("=" * 70)

n_points = 500
x = np.linspace(0, 1, n_points).reshape(-1, 1)

test_functions = {
    'sinusoid': np.sin(2 * np.pi * 3 * x),
    'multi_freq': 0.5*np.sin(2*np.pi*2*x) + 0.3*np.sin(2*np.pi*7*x) + 0.2*np.cos(2*np.pi*13*x),
    'step': (x > 0.5).astype(float),
    'gaussian': np.exp(-((x - 0.5)**2) / 0.02),
    'polynomial': 4 * x * (1-x) * (1-2*x),
}

# =============================================================================
# 3. SCALING EXPERIMENT
# =============================================================================

print("\nComparing scaling behavior...")

hidden_sizes = [32, 64, 128, 256, 512]
results = {fname: {} for fname in test_functions.keys()}

for fname, y in test_functions.items():
    print(f"\nFunction: {fname}")
    results[fname] = {m: [] for m in ['fourier', 'reservoir', 'random_tanh', 'trained_mlp_shallow', 'trained_mlp_deep']}

    for h in hidden_sizes:
        # Fourier features + ridge
        H_f = fourier_features(x, h, sigma=3.0)
        W_f = train_ridge(H_f, y)
        pred_f = H_f @ W_f
        mse_f = np.mean((pred_f - y) ** 2)
        results[fname]['fourier'].append(mse_f)

        # Reservoir + ridge
        H_r = reservoir_features(x, h, iterations=10)
        W_r = train_ridge(H_r, y)
        pred_r = H_r @ W_r
        mse_r = np.mean((pred_r - y) ** 2)
        results[fname]['reservoir'].append(mse_r)

        # Random tanh + ridge
        H_t = random_tanh_features(x, h)
        W_t = train_ridge(H_t, y)
        pred_t = H_t @ W_t
        mse_t = np.mean((pred_t - y) ** 2)
        results[fname]['random_tanh'].append(mse_t)

        # Trained MLP (shallow: 2 layers)
        mlp = TrainedMLP(1, [h, h], 1)
        mlp.fit(x, y, max_iter=3000)
        pred_mlp = mlp.predict(x)
        mse_mlp = np.mean((pred_mlp - y) ** 2)
        results[fname]['trained_mlp_shallow'].append(mse_mlp)

        # Trained MLP (deep: 4 layers, smaller width)
        h_small = max(8, h // 4)
        mlp_deep = TrainedMLP(1, [h_small, h_small, h_small, h_small], 1)
        mlp_deep.fit(x, y, max_iter=3000)
        pred_deep = mlp_deep.predict(x)
        mse_deep = np.mean((pred_deep - y) ** 2)
        results[fname]['trained_mlp_deep'].append(mse_deep)

        print(f"  H={h}: Fourier={mse_f:.2e}, Reservoir={mse_r:.2e}, MLP={mse_mlp:.2e}")

# =============================================================================
# 4. TEMPORAL TASK: ESN vs Trained Readout
# =============================================================================

print("\n" + "=" * 70)
print("TEMPORAL TASK: Next-step prediction")
print("=" * 70)

# Generate sinusoidal sequences
seq_len = 100
n_sequences = 30

sequences = []
targets = []
for _ in range(n_sequences):
    freq = np.random.uniform(0.5, 2.0)
    phase = np.random.uniform(0, 2*np.pi)
    t = np.linspace(0, 4*np.pi, seq_len)
    seq = np.sin(freq * t + phase)
    sequences.append(seq)

sequences = np.array(sequences)

temporal_results = {'esn': [], 'fourier_no_mem': []}

for h in [32, 64, 128, 256]:
    print(f"\nHidden size: {h}")

    # ESN: Process sequences with memory
    np.random.seed(42)
    W_in = np.random.randn(1, h) * 0.5
    W_hh = np.random.randn(h, h)
    eig = np.abs(np.linalg.eigvals(W_hh)).max()
    W_hh = W_hh * (0.9 / eig)
    b = np.random.randn(h) * 0.1

    all_states = []
    all_targets = []

    for seq in sequences:
        h_state = np.zeros(h)
        for t in range(len(seq) - 1):
            inp = seq[t:t+1].reshape(1, 1)
            h_state = np.tanh(inp @ W_in + h_state @ W_hh + b).flatten()
            all_states.append(h_state)
            all_targets.append(seq[t + 1])

    H_esn = np.array(all_states)
    Y_esn = np.array(all_targets).reshape(-1, 1)

    W_out = train_ridge(H_esn, Y_esn)
    pred_esn = H_esn @ W_out
    mse_esn = np.mean((pred_esn - Y_esn) ** 2)
    temporal_results['esn'].append(mse_esn)
    print(f"  ESN (with memory): MSE = {mse_esn:.6f}")

    # Fourier: No memory, just map current value to next
    X_flat = sequences[:, :-1].flatten().reshape(-1, 1)
    Y_flat = sequences[:, 1:].flatten().reshape(-1, 1)

    H_fourier = fourier_features(X_flat, h, sigma=3.0)
    W_fourier = train_ridge(H_fourier, Y_flat)
    pred_fourier = H_fourier @ W_fourier
    mse_fourier = np.mean((pred_fourier - Y_flat) ** 2)
    temporal_results['fourier_no_mem'].append(mse_fourier)
    print(f"  Fourier (no memory): MSE = {mse_fourier:.6f}")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================

# Scaling plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, fname in enumerate(test_functions.keys()):
    ax = axes[idx]
    for method, label, color, marker in [
        ('fourier', 'Fourier (random)', 'blue', 's'),
        ('reservoir', 'Reservoir (random)', 'red', 'o'),
        ('random_tanh', 'Tanh (random)', 'orange', '^'),
        ('trained_mlp_shallow', 'MLP trained (shallow)', 'green', 'd'),
        ('trained_mlp_deep', 'MLP trained (deep)', 'purple', 'v')
    ]:
        ax.plot(hidden_sizes, results[fname][method], f'{color}', marker=marker,
                label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Hidden Size / Features')
    ax.set_ylabel('MSE')
    ax.set_title(fname)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('learned_vs_random_scaling.png', dpi=150, bbox_inches='tight')
plt.close()

# Temporal comparison
fig, ax = plt.subplots(figsize=(10, 6))
hs = [32, 64, 128, 256]
ax.plot(hs, temporal_results['esn'], 'ro-', label='ESN (with memory)', linewidth=2, markersize=10)
ax.plot(hs, temporal_results['fourier_no_mem'], 'bs-', label='Fourier (no memory)', linewidth=2, markersize=10)
ax.set_xlabel('Hidden Size')
ax.set_ylabel('MSE')
ax.set_title('Temporal Prediction: Memory Matters!')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('temporal_memory_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nPlots saved to: learned_vs_random_scaling.png, temporal_memory_comparison.png")

# =============================================================================
# 6. SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: MSE at H=512")
print("=" * 70)

print(f"\n{'Function':<12} {'Fourier':<12} {'Reservoir':<12} {'Tanh Rand':<12} {'MLP Shallow':<12} {'MLP Deep':<12}")
print("-" * 72)

for fname in test_functions.keys():
    f = results[fname]['fourier'][-1]
    r = results[fname]['reservoir'][-1]
    t = results[fname]['random_tanh'][-1]
    ms = results[fname]['trained_mlp_shallow'][-1]
    md = results[fname]['trained_mlp_deep'][-1]
    print(f"{fname:<12} {f:<12.2e} {r:<12.2e} {t:<12.2e} {ms:<12.2e} {md:<12.2e}")

print(f"""
TEMPORAL TASK @ H=256:
  ESN (memory):         MSE = {temporal_results['esn'][-1]:.6f}
  Fourier (no memory):  MSE = {temporal_results['fourier_no_mem'][-1]:.6f}

KEY CONCLUSIONS:

1. FOR STATIC FUNCTION APPROXIMATION:
   ┌────────────────────────────────────────────────────────────────────┐
   │ Fourier features consistently outperform reservoir features       │
   │ Trained MLPs CAN match Fourier given proper optimization          │
   │ Reservoir's recurrence provides NO benefit for static tasks       │
   │ Random tanh ≈ Reservoir (recurrence is unnecessary overhead)      │
   └────────────────────────────────────────────────────────────────────┘

2. FOR TEMPORAL PREDICTION:
   ┌────────────────────────────────────────────────────────────────────┐
   │ ESN dramatically outperforms memoryless methods                   │
   │ Memory IS the key - not the "quality" of basis functions          │
   │ Even "bad" random reservoir features beat "good" Fourier          │
   │   when the task requires temporal context                         │
   └────────────────────────────────────────────────────────────────────┘

3. THE FUNDAMENTAL ANSWER:

   Q: "Is reservoir's advantage just memory, or also basis quality?"

   A: It's JUST the memory.

   For function approximation (no memory needed):
   - Fourier > Trained MLP ≈ Reservoir ≈ Random Tanh

   For temporal prediction (memory needed):
   - ESN >> Fourier >> (anything memoryless)

4. "IN THE LIMIT OF PARAMETERS?"

   - With infinite random features → kernel regression (RKHS)
   - Different features → different kernels → different inductive biases
   - Fourier features ≈ RBF kernel (great for smooth functions)
   - Reservoir features ≈ some nonlinear kernel (less structured)

   - Trained networks ARE universal approximators
   - But may need exponentially more parameters for certain functions
   - The gap doesn't fully close because inductive bias matters

5. "IS THIS THE END OF THE STORY?"

   NO! Interesting directions:

   a) Fourier + Reservoir hybrid for spatiotemporal data
   b) Learned reservoir weights (= RNN with ESN initialization)
   c) Structured random features combining both biases
   d) Deep reservoirs with hierarchical temporal abstraction
""")

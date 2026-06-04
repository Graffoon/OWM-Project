import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =============================================================================
# OWM Projector Demo
# Visualises how the protection matrix P evolves across training batches,
# and how the influence of new inputs becomes increasingly redundant over time.
# Uses a 3x3 example for clarity — real networks use (hidden_dim x hidden_dim).
# =============================================================================

ALPHA = 0.75       # Regularisation scalar — controls how aggressively P tightens
N_BATCHES = 40     # Number of batches to simulate
np.random.seed(42) # Fixed seed so the demo is reproducible


def update_projector(P, x_mean, alpha):
    """
    RLS update of the OWM projector matrix P (Eq. 2, Zeng et al. 2019).
    Takes one step of closing off the direction of x_mean in P.
    Returns the updated P and the xPx scalar (used to track influence over time).
    """
    x_mean      = x_mean.reshape(-1, 1)        # Ensure column vector shape (3, 1)
    Px          = P @ x_mean                   # How much of x_mean passes through P
    xPx         = (x_mean.T @ Px).item()       # Scalar: input influence on current subspace
    denominator = alpha + xPx                  # Stabilised denominator
    numerator   = Px @ Px.T                    # Rank-1 correction matrix
    P_new       = P - numerator / denominator  # Downdate: close off x_mean's direction
    return P_new, xPx


# =============================================================================
# Simulation
# =============================================================================

P = np.eye(3)                        # Start as full identity — no directions blocked yet
P_history    = [P.copy()]            # Stores P snapshot after every batch
xPx_history  = []                    # Tracks how much influence each batch's input had
diag_history = [np.diag(P).copy()]   # Tracks the three diagonal values of P

# Generate random unit-normalised batch mean inputs to simulate varied training data
batch_inputs = [np.random.randn(3) for _ in range(N_BATCHES)]
batch_inputs = [x / np.linalg.norm(x) for x in batch_inputs]  # Normalise to unit length

for x_mean in batch_inputs:
    P, xPx = update_projector(P, x_mean, ALPHA)  # Apply one projector update
    P_history.append(P.copy())                    # Save P state after this batch
    xPx_history.append(xPx)                       # Save influence scalar
    diag_history.append(np.diag(P).copy())        # Save diagonal values


# =============================================================================
# Plotting
# =============================================================================

fig = plt.figure(figsize=(18, 11))

# Two-row layout: top row has 5 columns (4 heatmaps + colorbar), bottom row has 2 charts
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.5)

# --- Top row: P matrix heatmaps at four checkpoints -------------------------
checkpoints = [0, 1, 5, 10]
titles      = ['P — Start (Identity)', 'P — After Batch 1',
               'P — After Batch 5',    'P — After Batch 10']

axes_heatmaps = []
for plot_idx, (batch_idx, title) in enumerate(zip(checkpoints, titles)):
    ax = fig.add_subplot(gs[0, plot_idx])   # One heatmap per column
    axes_heatmaps.append(ax)
    snapshot = P_history[batch_idx]
    im = ax.imshow(snapshot, vmin=-1, vmax=1, cmap='RdBu')  # Red = negative, Blue = positive
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['x', 'y', 'z'])
    ax.set_yticklabels(['x', 'y', 'z'])

    # Annotate each cell with its value
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{snapshot[i, j]:.2f}",
                    ha='center', va='center', fontsize=8,
                    color='white' if abs(snapshot[i, j]) > 0.5 else 'black')

# Dedicated 5th column just for the colorbar — prevents it overlapping the heatmaps
cbar_ax = fig.add_subplot(gs[0, 4])
cbar_ax.set_visible(False)                         # Hide the axes itself
cbar = fig.colorbar(im, ax=cbar_ax, fraction=0.8,  # Attach colorbar to the invisible axes
                    pad=0.05, label='Matrix value')

# --- Bottom left: diagonal values over time ----------------------------------
ax_diag = fig.add_subplot(gs[1, :3])   # Spans first three columns
diag_array = np.array(diag_history)    # Shape (N_BATCHES+1, 3)
ax_diag.plot(diag_array[:, 0], color='#E74C3C', marker='.', markersize=3, label='P[x,x]')
ax_diag.plot(diag_array[:, 1], color='#2ECC71', marker='.', markersize=3, label='P[y,y]')
ax_diag.plot(diag_array[:, 2], color='#3498DB', marker='.', markersize=3, label='P[z,z]')
ax_diag.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax_diag.set_xlabel('Batch')
ax_diag.set_ylabel('Diagonal value')
ax_diag.set_title('P Diagonal Over Time\n(1.0 = direction fully open,  ~0 = fully blocked)',
                  fontsize=10, fontweight='bold')
ax_diag.legend()
ax_diag.set_xlim(0, N_BATCHES)

# --- Bottom right: input influence (xPx) over time --------------------------
ax_infl = fig.add_subplot(gs[1, 3:])   # Spans last two columns
ax_infl.plot(range(1, N_BATCHES + 1), xPx_history,
             color='#9B59B6', marker='.', markersize=3, label='xPx (input influence)')
ax_infl.fill_between(range(1, N_BATCHES + 1), xPx_history,
                     alpha=0.15, color='#9B59B6')  # Shaded area under curve
ax_infl.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax_infl.set_xlabel('Batch')
ax_infl.set_ylabel('xPx scalar')
ax_infl.set_title('Input Influence on P Over Time\n(high = input can still shift P,  low = subspace saturated)',
                  fontsize=10, fontweight='bold')
ax_infl.legend()
ax_infl.set_xlim(1, N_BATCHES)

fig.suptitle(
    'OWM Projector Matrix P — How Task Protection Builds Up Over Training\n'
    '(3×3 example; real networks use hidden_dim × hidden_dim)',
    fontsize=12, fontweight='bold'
)

plt.savefig('projector_demo.png', dpi=150, bbox_inches='tight')
print("Figure saved → projector_demo.png")
plt.show()

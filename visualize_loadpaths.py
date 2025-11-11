import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Plot all P_ij vs F11 for uniaxial and biaxial load paths
# -----------------------------------------------------------
def plot_all_pij_vs_f11(named_sets, cases=("uniaxial","biaxial")):
    """
    Creates 9 subplots (3x3) for each of the specified test cases.
    Each subplot shows one stress component P_ij versus F11.
    
    named_sets : dict
        Dictionary like data["calibration"] containing test cases.
    cases : tuple of str
        Names of the load paths to visualize (keys in named_sets).
    """
    for case in cases:
        if case not in named_sets:
            print(f"⚠️  Case '{case}' not found in dataset.")
            continue

        D = named_sets[case]
        F = np.array([s.F for s in D])
        P = np.array([s.P for s in D])
        F11 = F[:, 0, 0]

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"Stress Components vs F11 — {case}", fontsize=14, y=0.93)

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                ax.plot(F11, P[:, i, j], marker='o', ms=3, label=f"P{i+1}{j+1}")
                ax.set_xlabel("F11")
                ax.set_ylabel(f"P{i+1}{j+1}")
                ax.grid(True)
                ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        plt.show()

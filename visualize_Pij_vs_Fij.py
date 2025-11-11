import numpy as np
import matplotlib.pyplot as plt

def plot_pij_vs_fij_separate(named_sets, cases=("uniaxial", "biaxial"), save=False, outdir="plots"):
    """
    For each case in `cases`, creates a 3x3 subplot figure where each subplot
    shows P_ij versus F_ij (i,j in {1,2,3}).

    Parameters
    ----------
    named_sets : dict
        e.g., data["calibration"] (keys should include "uniaxial" and/or "biaxial").
    cases : tuple
        Which load paths to plot.
    save : bool
        If True, saves PNGs to `outdir`.
    outdir : str
        Directory to save figures if save=True.
    """
    for case in cases:
        if case not in named_sets:
            print(f"⚠️  Case '{case}' not found in dataset.")
            continue

        D = named_sets[case]
        F = np.array([s.F for s in D])   # (N,3,3)
        P = np.array([s.P for s in D])   # (N,3,3)

        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        fig.suptitle(f"P_ij vs F_ij — {case.capitalize()} load", fontsize=14, y=0.93)

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                fij = F[:, i, j]
                pij = P[:, i, j]
                ax.plot(fij, pij, marker='o', ms=3, linestyle='-', label=f"P{i+1}{j+1} vs F{i+1}{j+1}")
                ax.set_xlabel(f"F{i+1}{j+1}")
                ax.set_ylabel(f"P{i+1}{j+1}")
                ax.grid(True)
                ax.legend(fontsize=8, loc='best')

        plt.tight_layout()
        if save:
            import os
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(f"{outdir}/{case}_Pij_vs_Fij.png", dpi=200, bbox_inches="tight")
        plt.show()

        

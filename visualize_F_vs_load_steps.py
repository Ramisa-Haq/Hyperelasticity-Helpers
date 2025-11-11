import numpy as np
import matplotlib.pyplot as plt

def plot_F_vs_steps(named_sets, cases=("uniaxial", "biaxial"), components=((0,0),), save=False, outdir="plots"):
    """
    Plots selected F_ij components vs load steps for the given load cases.

    Parameters
    ----------
    named_sets : dict
        e.g., data["calibration"] or data["test"], containing test cases.
    cases : tuple of str
        Load path names to visualize (keys in named_sets).
    components : tuple of (i,j)
        Indices of F components to plot (0-based indexing, e.g., (0,0) for F11).
        Example: components=((0,0),(1,1),(2,2)) plots F11, F22, F33.
    save : bool
        If True, saves figures as PNGs.
    outdir : str
        Directory to save figures if save=True.
    """

    for case in cases:
        if case not in named_sets:
            print(f"⚠️  Case '{case}' not found in dataset.")
            continue

        D = named_sets[case]
        F = np.array([s.F for s in D])   # shape (N, 3, 3)
        steps = np.arange(len(F))

        fig, ax = plt.subplots(figsize=(7, 5))
        for (i, j) in components:
            if i > 2 or j > 2:
                print(f"⚠️ Invalid index ({i},{j}). Must be between 0 and 2.")
                continue
            label = f"F{i+1}{j+1}"
            ax.plot(steps, F[:, i, j], marker='o', ms=3, linestyle='-', label=label)

        ax.set_xlabel("Load Step")
        ax.set_ylabel("F component")
        ax.set_title(f"Selected F components vs Load Steps — {case.capitalize()} Load")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if save:
            import os
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(f"{outdir}/{case}_selected_F_vs_steps.png", dpi=200, bbox_inches="tight")
        plt.show()

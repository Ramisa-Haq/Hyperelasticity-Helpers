import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# P11 vs F11 — stress–stretch behavior for each test
# -----------------------------------------------------------
def plot_p11_vs_f11(named_sets, title="P11 vs F11", ax=None):
    """
    Plots nominal stress (P11) vs stretch (F11) for all load paths in a dataset.
    named_sets : dict
        Dictionary like data["calibration"] or data["test"]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    for name, D in named_sets.items():
        F = np.array([s.F for s in D])
        P = np.array([s.P for s in D])
        ax.plot(F[:, 0, 0], P[:, 0, 0], label=name)

    ax.set_xlabel("F11 (stretch)")
    ax.set_ylabel("P11 (nominal stress)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax

# -----------------------------------------------------------
# F11, F22, F33 vs sample index — deformation components
# -----------------------------------------------------------
def plot_deformation_components(D, name="test_case", title="Deformation Components", ax=None):
    """
    Plots F11, F22, F33 vs sample index for one load path.
    D : list of Samples for one test (e.g., data["calibration"]["uniaxial"])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    F = np.array([s.F for s in D])
    ax.plot(F[:, 0, 0], label="F11")
    ax.plot(F[:, 1, 1], label="F22")
    ax.plot(F[:, 2, 2], label="F33")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Deformation gradient component")
    ax.set_title(f"{title}: {name}")
    ax.legend()
    ax.grid(True)
    return ax

# -----------------------------------------------------------
# F11 vs F22 — deformation space (useful to compare load paths)
# -----------------------------------------------------------
def plot_deformation_space(calib_sets, test_sets=None, ax=None):
    """
    Plots F11 vs F22 for calibration and test datasets to visualize load paths.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # Calibration load paths
    for name, D in calib_sets.items():
        F = np.array([s.F for s in D])
        ax.plot(F[:, 0, 0], F[:, 1, 1], label=f"calib: {name}", lw=2)

    # Optional: Test load paths (dashed)
    if test_sets:
        for name, D in test_sets.items():
            F = np.array([s.F for s in D])
            ax.plot(F[:, 0, 0], F[:, 1, 1], '--', label=f"test: {name}")

    ax.set_xlabel("F11")
    ax.set_ylabel("F22")
    ax.set_title("Deformation Space: Calibration vs Test")
    ax.legend()
    ax.grid(True)
    return ax

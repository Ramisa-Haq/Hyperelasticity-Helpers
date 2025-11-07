import numpy as np
from pathlib import Path # Path: from pathlib module, used for convenient and safe file path handling
from collections import namedtuple

Sample = namedtuple("Sample", ["F", "P", "W"])

def load_txt_file(path):
    X = np.loadtxt(path)               # shape: [N, 19]
    F = X[:, :9].reshape(-1, 3, 3)     # F11..F33 → (N,3,3)
    P = X[:, 9:18].reshape(-1, 3, 3)   # P11..P33 → (N,3,3)
    W = X[:, 18]                       # (N,)
    D = [Sample(F[i], P[i], W[i]) for i in range(len(W))]
    return D
# D = load_txt_file(Path("hyperelasticity/data/calibration/uniaxial.txt"))
# print(len(D))

def load_folder(folder):
    folder = Path(folder)
    out = {}
    for txt in sorted(folder.glob("*.txt")):
        out[txt.stem] = load_txt_file(txt)
    return out

#split（calibration/test）→ {'calibration': {...}, 'test': {...}}
def load_split(root="hyperelasticity/data"):
    root = Path(root)
    data = {
        "calibration": load_folder(root / "calibration"),
        "test":        load_folder(root / "test"),
    }
    return data

# def combine_sets(named_sets):
#     D = []
#     for samples in named_sets.values():
#         D.extend(samples)
#     return D

def summarize(split):
    for split_name, sets in split.items():
        total = sum(len(v) for v in sets.values())
        print(f"[{split_name}] total samples = {total}")
        for name, samples in sets.items():
            print(f"  - {name}: {len(samples)}")


def plot_F11_P11(named_sets, ax=None, title=""):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,4))
    for name, D in named_sets.items():
        F11 = [s.F[0,0] for s in D]
        P11 = [s.P[0,0] for s in D]
        ax.plot(F11, P11, label=name)
    ax.set_xlabel("F11"); ax.set_ylabel("P11"); ax.set_title(title); ax.legend()
    return ax

data = load_split("hyperelasticity/data")
summarize(data)


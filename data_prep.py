import numpy as np

def compute_C(F):
    return F.T @ F

def C_vec(C):
    return np.array([C[0,0], C[1,1], C[2,2], C[0,1], C[0,2], C[1,2]])

def P_vec(P):
    return P.flatten()  # (9,)

def prepare_data(data):
    """Extract (X, Y) from loaded data structure—combining all calibration sets."""
    X_list = []
    Y_list = []
    for load_path_samples in data["calibration"].values():
        X_list += [C_vec(compute_C(s.F)) for s in load_path_samples]
        Y_list += [P_vec(s.P) for s in load_path_samples]
    X = np.stack(X_list)
    Y = np.stack(Y_list)
    return X, Y

# For a specific set (e.g., "uniaxial"), do:
def prepare_single_path(samples):
    X = np.stack([C_vec(compute_C(s.F)) for s in samples])
    Y = np.stack([P_vec(s.P) for s in samples])
    return X, Y

# Usage:
# samples = data["calibration"]["uniaxial"]
# X, Y = prepare_single_path(samples)

# Or, for all calibration data:
# X, Y = prepare_data(data)

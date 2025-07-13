import numpy as np
from typing import Callable, List, Set
from FairOT.sinkhorn import pot_partial_extended, pot_partial_library
import matplotlib.pyplot as plt

def greedy_fair_prototype_selection_with_obj(f: Callable, S: np.ndarray, k: int, reg: float) -> (List[int], List[float]):
    """
    Greedy algorithm with objective tracking for fair prototype selection.
    Returns selected indices and list of f(P) values at each step.
    """
    n = S.shape[0]
    P = []
    candidates = set(range(n))
    obj_values = []
    gamma_P = None
    for i in range(k):
        # Solve partial OT for current P
        if len(P) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P, range(n))]
            mu_P = np.ones(len(P)) / len(P)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values.append(obj_P)
        # Greedy selection via approximate gain
        best_gain = -np.inf
        best_v = None
        for v in candidates - set(P):
            gain = f(P, gamma_P, v, S, k, reg)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        P.append(best_v)
    return P, obj_values

# Example usage
def main():
    n = 100
    k = 15
    reg = 0.05
    # Generate random 2D points
    X = np.random.randn(n, 2)
    # Gaussian similarity matrix
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)

    # --- Approx-gain greedy (pot_partial_extended) ---
    P_approx = []
    obj_values_approx = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx, range(n))]
            mu_P = np.ones(len(P_approx)) / len(P_approx)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_approx.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx):
            approx = approx_gain(P_approx, gamma_P, v, S, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx.append(best_v)
    S_P = S[np.ix_(P_approx, range(n))]
    mu_P = np.ones(len(P_approx)) / len(P_approx)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_approx.append(obj_P)

    # --- Actual-gain greedy (pot_partial_extended) ---
    P_actual = []
    obj_values_actual = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual, range(n))]
            mu_P = np.ones(len(P_actual)) / len(P_actual)
            gamma_P, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
        obj_values_actual.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual):
            P_new = P_actual + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual.append(best_v)
    S_P = S[np.ix_(P_actual, range(n))]
    mu_P = np.ones(len(P_actual)) / len(P_actual)
    _, obj_P = pot_partial_extended(S_P, k, mu_P, reg)
    obj_values_actual.append(obj_P)

    # --- Approx-gain greedy (pot_partial_library) ---
    P_approx_lib = []
    obj_values_approx_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_approx_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_approx_lib, range(n))]
            mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_approx_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_approx_lib):
            approx = approx_gain(P_approx_lib, gamma_P, v, S, k, reg)
            if approx > best_gain:
                best_gain = approx
                best_v = v
        P_approx_lib.append(best_v)
    S_P = S[np.ix_(P_approx_lib, range(n))]
    mu_P = np.ones(len(P_approx_lib)) / len(P_approx_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_approx_lib.append(obj_P)

    # --- Actual-gain greedy (pot_partial_library) ---
    P_actual_lib = []
    obj_values_actual_lib = []
    gamma_P = None
    obj_P = 0.0
    for step in range(k):
        if len(P_actual_lib) == 0:
            gamma_P = None
            obj_P = 0.0
        else:
            S_P = S[np.ix_(P_actual_lib, range(n))]
            mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
            gamma_P, obj_P = pot_partial_library(S_P, k, mu_P, reg)
        obj_values_actual_lib.append(obj_P)
        best_gain = -np.inf
        best_v = None
        for v in set(range(n)) - set(P_actual_lib):
            P_new = P_actual_lib + [v]
            S_P_new = S[np.ix_(P_new, range(n))]
            mu_P_new = np.ones(len(P_new)) / len(P_new)
            _, obj_P_new = pot_partial_library(S_P_new, k, mu_P_new, reg)
            actual = obj_P_new - obj_P
            if actual > best_gain:
                best_gain = actual
                best_v = v
        P_actual_lib.append(best_v)
    S_P = S[np.ix_(P_actual_lib, range(n))]
    mu_P = np.ones(len(P_actual_lib)) / len(P_actual_lib)
    _, obj_P = pot_partial_library(S_P, k, mu_P, reg)
    obj_values_actual_lib.append(obj_P)

    # --- Plot all four curves on the same plot ---
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, obj_values_approx, marker='o', label='Approx-gain (extended)')
    plt.plot(steps, obj_values_actual, marker='s', color='orange', label='Actual-gain (extended)')
    plt.plot(steps, obj_values_approx_lib, marker='^', color='green', label='Approx-gain (library)')
    plt.plot(steps, obj_values_actual_lib, marker='d', color='red', label='Actual-gain (library)')
    plt.title('Objective value: all greedy strategies')
    plt.xlabel('Greedy step (k)')
    plt.ylabel('Objective value f(P)')
    plt.grid(True)
    plt.legend()
    plt.xticks(steps)
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test_optimal_alpha_constraints()
    main()

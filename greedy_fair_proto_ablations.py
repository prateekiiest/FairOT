import numpy as np
from typing import Callable, List, Set
from FairOT.sinkhorn import pot_partial_extended, pot_partial_library
import matplotlib.pyplot as plt

def greedy_fair_prototype_selection(f: Callable, S: np.ndarray, k: int, reg: float) -> List[int]:
    """
    Greedy algorithm with approximate gain for fair prototype selection.
    Args:
        f: Function to compute gain (approximate gain function)
        S: Similarity matrix (n x n)
        k: Cardinality constraint
        reg: Entropic regularization parameter
    Returns:
        List of selected prototype indices (P)
    """
    n = S.shape[0]
    P = []
    candidates = set(range(n))
    for i in range(k):
        # Solve partial OT for current P
        if len(P) == 0:
            gamma_P = None
        else:
            S_P = S[np.ix_(P, range(n))]
            mu_P = np.ones(len(P)) / len(P)
            gamma_P, _ = pot_partial_extended(S_P, k, mu_P, reg)
        # Greedy selection via approximate gain
        best_gain = -np.inf
        best_v = None
        for v in candidates - set(P):
            gain = f(P, gamma_P, v, S, k, reg)
            if gain > best_gain:
                best_gain = gain
                best_v = v
        P.append(best_v)
    return P

def optimal_alpha(S_a: np.ndarray, b: np.ndarray, reg: float, tol=1e-8, max_iter=100) -> np.ndarray:
    """
    Compute the optimal alpha for the entropic regularized, box-constrained problem.
    KKT analysis:
    - For 0 < alpha_i < b_i (interior):
        nu_i = 0, so beta = S_a[i] - reg * (1 + log(alpha_i)), so alpha_i = exp((S_a[i] - beta)/reg - 1)
    - For alpha_i = b_i (upper boundary):
        nu_i = S_a[i] - beta - reg * (1 + log(b_i)), so beta <= S_a[i] - reg * (1 + log(b_i))
    The solution is a truncated and normalized softmax, found by bisection on beta.
    Args:
        S_a: Similarity vector for candidate a (shape n,)
        b: Upper bound vector (shape n,)
        reg: Entropic regularization parameter (lambda)
    Returns:
        alpha: Optimal solution (shape n,)
    """
    n = S_a.shape[0]
    def softmax_with_beta(beta):
        # Interior: alpha_i = exp((S_a[i] - beta)/reg - 1)
        # Boundary: alpha_i = b_i if unconstrained > b_i
        x = np.exp((S_a - beta) / reg - 1)
        return np.minimum(x, b)
    # Bisection search for beta so that sum(alpha) = 1
    BETA_REG_MULTIPLIER = 100
    beta_low = np.min(S_a) - BETA_REG_MULTIPLIER * reg
    beta_high = np.max(S_a) + BETA_REG_MULTIPLIER * reg
    for _ in range(max_iter):
        beta = (beta_low + beta_high) / 2
        alpha = softmax_with_beta(beta)
        total = np.sum(alpha)
        # Check normalization constraint
        if abs(total - 1) < tol:
            break
        if total > 1:
            beta_low = beta
        else:
            beta_high = beta
    # After bisection, alpha satisfies:
    # - If 0 < alpha_i < b_i: alpha_i = exp((S_a[i] - beta)/reg - 1)
    # - If alpha_i = b_i: beta <= S_a[i] - reg * (1 + log(b_i))
    return alpha

def approx_gain(P: List[int], gamma_P, v: int, S: np.ndarray, k: int, reg: float) -> float:
    """
    Approximate gain function for greedy selection using feasible extension.
    Args:
        P: Current set of prototypes
        gamma_P: Current OT plan (can be None if P is empty)
        v: Candidate index to add
        S: Similarity matrix
        k: Cardinality constraint
        reg: Entropic regularization parameter
    Returns:
        Approximate gain of adding v to P
    """
    n = S.shape[0]
    if gamma_P is None or len(P) == 0:
        # If P is empty, just solve for {v}
        S_P_new = S[np.ix_([v], range(n))]
        mu_P_new = np.ones(1)
        gamma_P_new, obj_new = pot_partial_extended(S_P_new, k, mu_P_new, reg)
        obj_old = 0.0
        return obj_new - obj_old
    else:
        m = len(P)
        S_P = S[np.ix_(P, range(n))]
        S_a = S[v, :].reshape(1, n)
        col_sums = np.sum(gamma_P, axis=0)
        mu_T = k * np.ones(n) / n
        b = mu_T - col_sums
        b = np.clip(b, 0, None)  # ensure non-negative upper bounds
        # Use closed-form for optimal alpha
        alpha = optimal_alpha(S_a.flatten(), b, reg)
        gamma_tilde = np.vstack([gamma_P, alpha.reshape(1, n)])
        obj = np.sum(S_P * gamma_P) + np.sum(S_a * alpha)
        mask = gamma_tilde > 0
        entropy = -np.sum(gamma_tilde[mask] * np.log(gamma_tilde[mask]))
        obj = obj + reg * entropy
        mask_old = gamma_P > 0
        entropy_old = -np.sum(gamma_P[mask_old] * np.log(gamma_P[mask_old]))
        obj_old = np.sum(S_P * gamma_P) + reg * entropy_old
        return obj - obj_old

def test_optimal_alpha_constraints():
    np.random.seed(42)
    n = 10
    k = 5
    reg = 0.1
    S = np.random.rand(n, n)
    S = (S + S.T) / 2
    np.fill_diagonal(S, 1.0)
    # Select a random set P of size m
    m = 3
    P = np.random.choice(n, m, replace=False).tolist()
    S_P = S[np.ix_(P, range(n))]
    mu_P = np.ones(m) / m
    gamma_P_star, _ = pot_partial_extended(S_P, k, mu_P, reg)
    # Pick a candidate v not in P
    v = np.random.choice(list(set(range(n)) - set(P)))
    S_a = S[v, :]
    mu_T = k * np.ones(n) / n
    col_sums = np.sum(gamma_P_star, axis=0)
    b = mu_T - col_sums
    b = np.clip(b, 0, None)
    alpha = optimal_alpha(S_a, b, reg)
    print("alpha:", alpha)
    print("sum(alpha):", np.sum(alpha))
    print("min(alpha):", np.min(alpha))
    print("max(alpha):", np.max(alpha))
    print("b:", b)
    # Check constraints with visual signals
    all_pass = True
    for i in range(n):
        interior = (alpha[i] < b[i] - 1e-8)
        boundary = (abs(alpha[i] - b[i]) < 1e-6)
        nonneg = (alpha[i] >= -1e-8)
        kkt_interior = False
        kkt_boundary = False
        if interior and nonneg:
            # KKT: alpha_i = exp((S_a[i] - beta)/reg - 1) for some beta
            kkt_interior = True
        if boundary and nonneg:
            # KKT: beta <= S_a[i] - reg * (1 + log(b[i]))
            kkt_boundary = True
        if interior and nonneg and kkt_interior:
            print(f"\u2705 alpha[{i}] interior OK: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
        elif boundary and nonneg and kkt_boundary:
            print(f"\u2705 alpha[{i}] boundary OK: {alpha[i]:.4f} == b[{i}]={b[i]:.4f}")
        elif not nonneg:
            print(f"\u274C alpha[{i}] NEGATIVE: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
            all_pass = False
        else:
            print(f"\u274C alpha[{i}] violates KKT: {alpha[i]:.4f} vs b[{i}]={b[i]:.4f}")
            all_pass = False
    if abs(np.sum(alpha) - 1) < 1e-6:
        print("\u2705 sum(alpha) == 1")
    else:
        print(f"\u274C sum(alpha) = {np.sum(alpha):.6f} (should be 1)")
        all_pass = False
    if all_pass:
        print("\u2705 All coordinate-wise KKT constraints satisfied.")
    else:
        print("\u274C Some constraints failed.")

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

import numpy as np
from typing import Callable, List, Set
from FairOT.sinkhorn import pot_partial_extended
import matplotlib.pyplot as plt

def greedy_fair_prototype_selection(f: Callable, S: np.ndarray, k: int, reg: float) -> List[int]:

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

    n = S_a.shape[0]
    beta_low = np.min(S_a) - 100 * reg
    beta_high = np.max(S_a) + 100 * reg
    for _ in range(max_iter):
        beta = (beta_low + beta_high) / 2
        alpha = np.minimum(np.exp((S_a - beta) / reg - 1),b)
        total = np.sum(alpha)
        if abs(total - 1) < tol:
            break
        if total > 1:
            beta_low = beta
        else:
            beta_high = beta
    return alpha

def approx_gain(P: List[int], gamma_P, v: int, S: np.ndarray, k: int, reg: float) -> float:
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
        b = np.clip(b, 0, None)  
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
    m = 3
    P = np.random.choice(n, m, replace=False).tolist()
    S_P = S[np.ix_(P, range(n))]
    mu_P = np.ones(m) / m
    gamma_P_star, _ = pot_partial_extended(S_P, k, mu_P, reg)
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
    all_pass = True
    for i in range(n):
        interior = (alpha[i] < b[i] - 1e-8)
        boundary = (abs(alpha[i] - b[i]) < 1e-6)
        nonneg = (alpha[i] >= -1e-8)
        kkt_interior = False
        kkt_boundary = False
        if interior and nonneg:
            kkt_interior = True
        if boundary and nonneg:
            kkt_boundary = True
        if interior and nonneg and kkt_interior:
            print(f"alpha[{i}] interior OK: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
        elif boundary and nonneg and kkt_boundary:
            print(f" alpha[{i}] boundary OK: {alpha[i]:.4f} == b[{i}]={b[i]:.4f}")
        elif not nonneg:
            print(f"alpha[{i}] NEGATIVE: {alpha[i]:.4f} < b[{i}]={b[i]:.4f}")
            all_pass = False
        else:
            print(f" alpha[{i}] violates KKT: {alpha[i]:.4f} vs b[{i}]={b[i]:.4f}")
            all_pass = False
    if abs(np.sum(alpha) - 1) < 1e-6:
        print(" sum(alpha) == 1")
    else:
        print(f" sum(alpha) = {np.sum(alpha):.6f} (should be 1)")
        all_pass = False
    if all_pass:
        print("All coordinate-wise KKT constraints satisfied.")
    else:
        print(" Some constraints failed.")

def greedy_fair_prototype_selection_with_obj(f: Callable, S: np.ndarray, k: int, reg: float) -> (List[int], List[float]):


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

def main():
    n = 100
    k = 15
    reg = 0.05
    X = np.random.randn(n, 2)
    # Gaussian similarity matrix
    sigma = 1.0
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    S = np.exp(-dists**2 / (2 * sigma**2))
    np.fill_diagonal(S, 1.0)

    #  Approx-gain greedy 
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

    # Actual-gain greedy 
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

    #  Plot both curves on the same plot 
    import matplotlib.ticker as mticker
    steps = range(1, k+2)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, obj_values_approx, marker='o', label='Approx-gain greedy')
    plt.plot(steps, obj_values_actual, marker='s', color='orange', label='Actual-gain greedy')
    plt.title('Objective value: approx-gain vs actual-gain greedy')
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

import numpy as np

# ===================================================
# Common setup
# ===================================================

STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}
rng = np.random.default_rng(12345)

def hamming2(a, b):
    return (a[0] ^ b[0]) + (a[1] ^ b[1])

# ===================================================
# Encoder branch outputs
# ===================================================

def branch_output_code1(prev_state_bit, input_bit):
    """Code-1: generators (1, 1+D)"""
    return (input_bit, input_bit ^ prev_state_bit)

def branch_output_code2(prev_state_bit, input_bit):
    """Code-2: generators (D, D+1)"""
    return (prev_state_bit, prev_state_bit ^ input_bit)

# ===================================================
# Viterbi update
# ===================================================

def viterbi_next_relative_metrics(rel_metrics, received, branch_output_func):
    M0, M1 = rel_metrics
    next_metrics_abs = [None, None]

    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state == 0 else M1
            out = branch_output_func(prev_state, next_state)
            d = hamming2(out, received)
            candidates.append(prev_metric + d)
        next_metrics_abs[next_state] = min(candidates)

    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0] - mmin), int(next_metrics_abs[1] - mmin))

    if rel_next not in STATE_INDEX:
        raise RuntimeError(f"Unexpected state {rel_next}")
    return rel_next

# ===================================================
# BSC and simulation
# ===================================================

def bsc_pair_sample(p):
    return (int(rng.random() < p), int(rng.random() < p))

def simulate_transition_matrix(p, branch_output_func, steps=200_000, burn_in=2000):
    counts = np.zeros((5, 5), dtype=np.int64)
    state = (0, 0)

    for _ in range(burn_in):
        r = bsc_pair_sample(p)
        state = viterbi_next_relative_metrics(state, r, branch_output_func)

    for _ in range(steps):
        i = STATE_INDEX[state]
        r = bsc_pair_sample(p)
        next_state = viterbi_next_relative_metrics(state, r, branch_output_func)
        j = STATE_INDEX[next_state]
        counts[i, j] += 1
        state = next_state

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

# ===================================================
# Theoretical matrix (Eq. 4) for Code-1
# ===================================================

def theoretical_matrix_code1(p):
    q = 1 - p
    T_paper = np.array([
        [q*q,    p*q,    0,        0,      p*p],
        [p*q,    0,      2*p*q,    0,      0],
        [0,      2*p*q,  0,        q*q+p*p,0],
        [0,      0,      p,        0,      0],
        [p*p,    0,      0,        0,      q*q]
    ])
    # Paper’s order: (2,0),(1,0),(0,0),(0,1),(0,2)
    paper_order = [(2,0),(1,0),(0,0),(0,1),(0,2)]
    reorder_idx = [paper_order.index(s) for s in STATE_LIST]
    return T_paper[reorder_idx,:][:,reorder_idx]

# ===================================================
# Main
# ===================================================

if __name__ == "__main__":
    # 1. Theoretical T for Code-1, p=0.5
    T_theory = theoretical_matrix_code1(0.5)
    print("=== Theoretical T (Code-1, Eq.4) at p=0.5 ===")
    for i,row in enumerate(T_theory):
        print(f"{STATE_LIST[i]} -> {np.round(row,4)}")

    # 2. Empirical T for Code-2 (D, D+1), p=0.1,0.2
    for p in [0.1, 0.2]:
        T_emp2 = simulate_transition_matrix(p, branch_output_code2)
        print(f"\n=== Empirical T_hat (Code-2, (D,D+1)) at p={p} ===")
        for i,row in enumerate(T_emp2):
            print(f"{STATE_LIST[i]} -> {np.round(row,4)}")

    # 3. Empirical T for Code-1 (1, 1+D), p=0.1,0.2
    for p in [0.1, 0.2]:
        T_emp1 = simulate_transition_matrix(p, branch_output_code1)
        print(f"\n=== Empirical T_hat (Code-1, (1,1+D)) at p={p} ===")
        for i,row in enumerate(T_emp1):
            print(f"{STATE_LIST[i]} -> {np.round(row,4)}")

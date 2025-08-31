import numpy as np

# ---------------------------
# State definitions
# ---------------------------

# Common state order
STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

def hamming2(a, b):
    return (a[0] ^ b[0]) + (a[1] ^ b[1])

# ---------------------------
# Encoder: (D, D+1)
# ---------------------------

def branch_output(prev_state_bit, input_bit):
    """
    Convolutional encoder with generators (D, D+1):
      output1 = u_{t-1}
      output2 = u_{t-1} xor u_t
    """
    return (prev_state_bit, prev_state_bit ^ input_bit)

# ---------------------------
# Viterbi update
# ---------------------------

def viterbi_next_relative_metrics(rel_metrics, received):
    M0, M1 = rel_metrics
    next_metrics_abs = [None, None]

    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state == 0 else M1
            out = branch_output(prev_state, next_state)
            d = hamming2(out, received)
            candidates.append(prev_metric + d)
        next_metrics_abs[next_state] = min(candidates)

    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0] - mmin), int(next_metrics_abs[1] - mmin))

    if rel_next not in STATE_INDEX:
        raise RuntimeError(f"Unexpected relative state encountered: {rel_next}")
    return rel_next

# ---------------------------
# Simulation of empirical transition matrix
# ---------------------------

rng = np.random.default_rng(12345)

def bsc_pair_sample(p):
    return (int(rng.random() < p), int(rng.random() < p))

def simulate_transition_matrix(p, steps=500_000, burn_in=2000):
    counts = np.zeros((5, 5), dtype=np.int64)
    state = (0, 0)

    for _ in range(burn_in):
        r = bsc_pair_sample(p)
        state = viterbi_next_relative_metrics(state, r)

    for _ in range(steps):
        i = STATE_INDEX[state]
        r = bsc_pair_sample(p)
        next_state = viterbi_next_relative_metrics(state, r)
        j = STATE_INDEX[next_state]
        counts[i, j] += 1
        state = next_state

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums

# ---------------------------
# Theoretical matrix from Code-1 (Eq. 4 of the paper)
# ---------------------------

def theoretical_matrix_code1(p):
    """
    Transition matrix for code with generators (1, 1+D),
    as given in Equation (4) of the paper.
    Ordered in the same state order: (0,2),(0,1),(0,0),(1,0),(2,0).
    """
    T = np.zeros((5,5))

    # For readability
    q = 1 - p

    # Eq. (4), but re-ordered into our state order
    # Paper's order: (2,0),(1,0),(0,0),(0,1),(0,2)
    # Our order:     (0,2),(0,1),(0,0),(1,0),(2,0)
    T_paper = np.array([
        [q*q,    p*q,    0,      0,    p*p],
        [p*q,    0,      2*p*q,  0,    0],
        [0,      2*p*q,  0,      q*q+p*p,  0],
        [0,      0,      p,      0,    0],
        [p*p,    0,      0,      0,    q*q]
    ])

    paper_order = [(2,0),(1,0),(0,0),(0,1),(0,2)]
    reorder_idx = [paper_order.index(s) for s in STATE_LIST]
    T = T_paper[reorder_idx,:][:,reorder_idx]
    return T

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Sweep over a range of p values to find best match
    ps = np.linspace(0.01, 0.49, 10)  # 10 values between 0.01 and 0.49
    best_p, best_diff = None, float("inf")

    for p in ps:
        T_emp = simulate_transition_matrix(p, steps=200_000)
        T_theory = theoretical_matrix_code1(p)

        diff = np.max(np.abs(T_emp - T_theory))

        print(f"\np = {p:.2f}")
        print("Empirical T (D, D+1):")
        for i,row in enumerate(T_emp):
            print(f"{STATE_LIST[i]} -> {np.round(row,4)}")

        print("\nTheoretical T (Code-1, Eq.4):")
        for i,row in enumerate(T_theory):
            print(f"{STATE_LIST[i]} -> {np.round(row,4)}")

        print(f"\nMax |Emp - Theory| = {diff:.6f}")

        if diff < best_diff:
            best_diff, best_p = diff, p

    print("\n=====================================")
    print(f"Best matching p ≈ {best_p:.3f} with max diff = {best_diff:.6f}")

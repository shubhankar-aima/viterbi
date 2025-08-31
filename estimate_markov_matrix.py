import numpy as np

# ---------------------------
# 1) Definitions & utilities
# ---------------------------

# New state order as requested
STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

def hamming2(a, b):
    """Hamming distance between two 2-bit tuples a=(a1,a2), b=(b1,b2)."""
    return (a[0] ^ b[0]) + (a[1] ^ b[1])

def branch_output(prev_state_bit, input_bit):
    """
    For [1, 1+D]: outputs (u_t, u_t xor u_{t-1})
    prev_state_bit = u_{t-1} in {0,1}
    input_bit      = u_t in {0,1}
    """
    return (input_bit, input_bit ^ prev_state_bit)

def viterbi_next_relative_metrics(rel_metrics, received):
    """
    One Viterbi step for the 2-state machine, using current relative metrics.
    rel_metrics = (M0, M1) with min(M0,M1)=0 (relative metrics).
    received    = (r1, r2) noisy channel outputs at this step.
    """
    M0, M1 = rel_metrics  # metrics for states s=0 and s=1

    next_metrics_abs = [None, None]  # metrics for next states 0 and 1

    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state == 0 else M1
            out = branch_output(prev_state, next_state)
            d = hamming2(out, received)
            candidates.append(prev_metric + d)
        next_metrics_abs[next_state] = min(candidates)

    # Convert to relative by subtracting the minimum
    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0] - mmin), int(next_metrics_abs[1] - mmin))

    if rel_next not in STATE_INDEX:
        raise RuntimeError(f"Unexpected relative state encountered: {rel_next}")
    return rel_next

def all_received_pairs_probs(p):
    """
    All possible received 2-bit pairs and their probabilities under BSC(p),
    given all-zero codeword is transmitted.
    """
    probs = {
        (0,0): (1-p)*(1-p),
        (0,1): (1-p)*p,
        (1,0): p*(1-p),
        (1,1): p*p
    }
    return list(probs.keys()), probs

# -------------------------------------------------------
# 2) Exact (by enumeration) transition matrix construction
# -------------------------------------------------------

def exact_transition_matrix(p):
    pairs, prob = all_received_pairs_probs(p)
    T = np.zeros((5, 5), dtype=float)

    for i, s in enumerate(STATE_LIST):
        for r in pairs:
            s_next = viterbi_next_relative_metrics(s, r)
            j = STATE_INDEX[s_next]
            T[i, j] += prob[r]
    return T

# -------------------------------------------------
# 3) Monte Carlo simulation of the transition matrix
# -------------------------------------------------

rng = np.random.default_rng(12345)

def bsc_pair_sample(p):
    """Sample received pair when (0,0) is transmitted."""
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
    T_hat = counts / row_sums
    return T_hat

# -----------------------
# 4) Main
# -----------------------

if __name__ == "__main__":
    p = float(input("Enter the crossover probability p (0 <= p <= 0.5): "))

    T_exact = exact_transition_matrix(p)
    T_emp = simulate_transition_matrix(p, steps=400_000)

    print("\nState order:", STATE_LIST)

    print("\nExact Transition Matrix (rows sum to 1):")
    for i, row in enumerate(T_exact):
        print(f"{STATE_LIST[i]} -> {row}")

    print("\nEmpirical Transition Matrix (from simulation):")
    for i, row in enumerate(T_emp):
        print(f"{STATE_LIST[i]} -> {np.round(row,6)}")

    diff = np.abs(T_emp - T_exact)
    print(f"\nMax absolute difference = {np.max(diff):.6g}")

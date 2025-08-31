import numpy as np

# ===================================================
# Config
# ===================================================
P = 0.1                 # p-value requested
STEPS = 5_000_000       # "many more iterations"
BURN_IN = 10_000
SEED = 123456789

# ===================================================
# Common state space (relative metric states)
# ===================================================
STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

# ===================================================
# Utilities
# ===================================================
def hamming2(a, b):
    """Hamming distance between two 2-bit tuples a=(a1,a2), b=(b1,b2)."""
    return (a[0] ^ b[0]) + (a[1] ^ b[1])

# ---------------------------
# Encoder branch outputs
# ---------------------------
def branch_output_code1(prev_state_bit, input_bit):
    """
    Code-1: (1, 1+D)
      y1 = u_t
      y2 = u_t xor u_{t-1}
    """
    return (input_bit, input_bit ^ prev_state_bit)

def branch_output_code2(prev_state_bit, input_bit):
    """
    Code-2: (D, D+1)
      y1 = u_{t-1}
      y2 = u_{t-1} xor u_t
    """
    return (prev_state_bit, prev_state_bit ^ input_bit)

# ---------------------------
# Viterbi one-step update using a chosen decoder trellis
# ---------------------------
def viterbi_next_relative_metrics(rel_metrics, received, decoder_branch_output):
    """
    rel_metrics = (M0, M1) with min=0, for trellis states {0,1} at time t.
    received    = (r1, r2) from the channel at time t.
    decoder_branch_output = function that produces hypothesized branch symbols
                            for the DECODER trellis (here: Code-1).
    Returns next relative metrics (M0', M1') with min subtracted.
    """
    M0, M1 = rel_metrics
    next_metrics_abs = [None, None]

    # Next state s' in {0,1} is the new input bit u_t for a rate-1/2, 2-state code
    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state == 0 else M1
            # Hypothesized symbols come from the DECODER trellis
            y_hat = decoder_branch_output(prev_state, next_state)
            d = hamming2(y_hat, received)
            candidates.append(prev_metric + d)
        next_metrics_abs[next_state] = min(candidates)

    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0] - mmin), int(next_metrics_abs[1] - mmin))

    if rel_next not in STATE_INDEX:
        raise RuntimeError(f"Unexpected relative state: {rel_next}")
    return rel_next

# ===================================================
# Channel model (transmit = Code-2, decode = Code-1)
# ===================================================
def make_rng(seed):
    return np.random.default_rng(seed)

def bsc_pair_from_code2_allzero(p, rng):
    """
    With all-zero input, Code-2's noiseless output is always (0,0).
    Pass (0,0) through BSC(p) independently per bit.
    """
    return (int(rng.random() < p), int(rng.random() < p))

def simulate_transition_matrix_mismatched(p, steps, burn_in, seed):
    """
    Transmitter: Code-2 (all-zero input => (0,0) each step), BSC(p).
    Decoder:     Code-1 Viterbi trellis for metric updates.
    Returns empirical 5x5 transition matrix in STATE_LIST order.
    """
    rng = make_rng(seed)
    counts = np.zeros((5, 5), dtype=np.int64)
    state = (0, 0)  # start in zero-relative-metric state

    # Burn-in
    for _ in range(burn_in):
        r = bsc_pair_from_code2_allzero(p, rng)
        state = viterbi_next_relative_metrics(state, r, decoder_branch_output=branch_output_code1)

    # Main simulation
    for _ in range(steps):
        i = STATE_INDEX[state]
        r = bsc_pair_from_code2_allzero(p, rng)
        next_state = viterbi_next_relative_metrics(state, r, decoder_branch_output=branch_output_code1)
        j = STATE_INDEX[next_state]
        counts[i, j] += 1
        state = next_state

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums, counts

# ===================================================
# Run
# ===================================================
if __name__ == "__main__":
    T_hat, counts = simulate_transition_matrix_mismatched(P, STEPS, BURN_IN, SEED)

    print(f"=== Empirical T_hat (Code-2 decoded with Code-1) at p={P} ===")
    print("State order:", STATE_LIST)
    for i, row in enumerate(T_hat):
        print(f"{STATE_LIST[i]} -> {np.round(row, 6)}")

    # Optional: show how many transitions were counted per row
    # (useful sanity check that rows have sufficient samples)
    # print("\nTransition counts per row:")
    # for i, row in enumerate(counts):
    #     print(f"{STATE_LIST[i]} -> {row.sum()} transitions")

import numpy as np

# ===================================================
# Config
# ===================================================
P = 0.1                 # channel crossover probability
STEPS = 500_000
BURN_IN = 5000
SEED = 12345

# ===================================================
# State definitions (relative metrics)
# ===================================================
STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

def hamming2(a, b):
    return (a[0] ^ b[0]) + (a[1] ^ b[1])

# ===================================================
# Encoder branch outputs
# ===================================================
def branch_output_code1(prev_state_bit, input_bit):
    # (1, 1+D)
    return (input_bit, input_bit ^ prev_state_bit)

def branch_output_code2(prev_state_bit, input_bit):
    # (D, D+1)
    return (prev_state_bit, prev_state_bit ^ input_bit)

# ===================================================
# Viterbi update
# ===================================================
def viterbi_next_relative_metrics(rel_metrics, received, decoder_branch_output):
    M0, M1 = rel_metrics
    next_metrics_abs = [None, None]

    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state == 0 else M1
            y_hat = decoder_branch_output(prev_state, next_state)
            d = hamming2(y_hat, received)
            candidates.append(prev_metric + d)
        next_metrics_abs[next_state] = min(candidates)

    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0] - mmin), int(next_metrics_abs[1] - mmin))

    if rel_next not in STATE_INDEX:
        raise RuntimeError(f"Unexpected state {rel_next}")
    return rel_next

# ===================================================
# Transmitter + channel
# ===================================================
def generate_received_pair(transmitter_code, input_bit, prev_state_bit, p, rng):
    """Generate a received 2-bit symbol given TX code, input, prev state, and BSC(p)."""
    if transmitter_code == 1:
        noiseless = branch_output_code1(prev_state_bit, input_bit)
    else:
        noiseless = branch_output_code2(prev_state_bit, input_bit)

    return (noiseless[0] ^ (rng.random() < p),
            noiseless[1] ^ (rng.random() < p)), noiseless

# ===================================================
# Simulation
# ===================================================
def simulate_transition_matrix(p, steps, burn_in,
                               transmitter_code, decoder_code,
                               input_bits=None, seed=1234, normalize=False):
    """
    transmitter_code ∈ {1,2}, decoder_code ∈ {1,2}
    input_bits = None (all-zero) or "random"
    normalize = if True, XOR received with true noiseless label before Viterbi
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((5,5), dtype=np.int64)
    state = (0,0)
    prev_state_bit = 0

    decoder_branch = branch_output_code1 if decoder_code==1 else branch_output_code2
    enc_func = branch_output_code1 if transmitter_code==1 else branch_output_code2

    # burn-in
    for _ in range(burn_in):
        input_bit = 0 if input_bits is None else int(rng.random()<0.5)
        r, noiseless = generate_received_pair(transmitter_code, input_bit, prev_state_bit, p, rng)
        r_use = (r[0]^noiseless[0], r[1]^noiseless[1]) if normalize else r
        state = viterbi_next_relative_metrics(state, r_use, decoder_branch)
        prev_state_bit = input_bit

    # main loop
    for _ in range(steps):
        input_bit = 0 if input_bits is None else int(rng.random()<0.5)
        r, noiseless = generate_received_pair(transmitter_code, input_bit, prev_state_bit, p, rng)
        r_use = (r[0]^noiseless[0], r[1]^noiseless[1]) if normalize else r
        i = STATE_INDEX[state]
        next_state = viterbi_next_relative_metrics(state, r_use, decoder_branch)
        j = STATE_INDEX[next_state]
        counts[i,j]+=1
        state = next_state
        prev_state_bit = input_bit

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums==0] = 1
    return counts/row_sums

# ===================================================
# Run all scenarios
# ===================================================
if __name__=="__main__":
    rng = np.random.default_rng(SEED)

    print("State order:", STATE_LIST)
    print("p-value: ", P)

    # # 1. Code-1 TX (all-zero) -> Code-1 RX
    # T1 = simulate_transition_matrix(P, STEPS, BURN_IN, 1, 1, input_bits=None, seed=SEED)
    # print("\n1. All-zero, Code-1→Code-1")
    # for i,row in enumerate(T1): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

    # # 2. Code-2 TX (all-zero) -> Code-1 RX
    # T2 = simulate_transition_matrix(P, STEPS, BURN_IN, 2, 1, input_bits=None, seed=SEED)
    # print("\n2. All-zero, Code-2→Code-1")
    # for i,row in enumerate(T2): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

    # # 3. Code-2 TX (all-zero) -> Code-2 RX
    # T3 = simulate_transition_matrix(P, STEPS, BURN_IN, 2, 2, input_bits=None, seed=SEED)
    # print("\n3. All-zero, Code-2→Code-2")
    # for i,row in enumerate(T3): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

    # 4. Code-1 TX (random, normalized) -> Code-1 RX
    T4 = simulate_transition_matrix(P, STEPS, BURN_IN, 1, 1, input_bits="random", seed=SEED, normalize=True)
    print("\n Random, Code-1→Code-1")
    for i,row in enumerate(T4): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

    # # 5. Code-2 TX (same random) -> Code-2 RX
    # T5 = simulate_transition_matrix(P, STEPS, BURN_IN, 2, 2, input_bits="random", seed=SEED)
    # print("\n5. Random, Code-2→Code-2")
    # for i,row in enumerate(T5): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

    # 6. Code-2 TX (same random) -> Code-1 RX
    T6 = simulate_transition_matrix(P, STEPS, BURN_IN, 2, 1, input_bits="random", seed=SEED)
    print("\n Random, Code-2→Code-1")
    for i,row in enumerate(T6): print(f"{STATE_LIST[i]} -> {np.round(row,2)}")

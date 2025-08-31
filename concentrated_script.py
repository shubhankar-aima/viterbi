import numpy as np

# ===================================================
# Config
# ===================================================
P1 = 0.1
P2 = 0.5
STEPS = 500_000
BURN_IN = 5000
SEED = 12345

STATE_LIST = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
STATE_INDEX = {s: i for i, s in enumerate(STATE_LIST)}

def hamming2(a, b):
    return (a[0]^b[0]) + (a[1]^b[1])

# Encoders
def branch_output_code1(prev_state_bit, input_bit):
    return (input_bit, input_bit ^ prev_state_bit)

def branch_output_code2(prev_state_bit, input_bit):
    return (prev_state_bit, prev_state_bit ^ input_bit)

# Viterbi update
def viterbi_next_relative_metrics(rel_metrics, received, decoder_branch_output):
    M0, M1 = rel_metrics
    next_metrics_abs = [None, None]
    for next_state in (0, 1):
        candidates = []
        for prev_state in (0, 1):
            prev_metric = M0 if prev_state==0 else M1
            y_hat = decoder_branch_output(prev_state, next_state)
            d = hamming2(y_hat, received)
            candidates.append(prev_metric+d)
        next_metrics_abs[next_state] = min(candidates)
    mmin = min(next_metrics_abs)
    rel_next = (int(next_metrics_abs[0]-mmin), int(next_metrics_abs[1]-mmin))
    if rel_next not in STATE_INDEX: raise RuntimeError(f"Unexpected state {rel_next}")
    return rel_next

# Channel
def bsc(p, pair, rng):
    return (pair[0] ^ (rng.random()<p), pair[1] ^ (rng.random()<p))

# Simulation with random input + coset normalization
def simulate_random_empirical(p, steps, burn_in, transmitter_code, decoder_code, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros((5,5),dtype=np.int64)
    state = (0,0); prev_state_bit=0
    decoder_branch = branch_output_code1 if decoder_code==1 else branch_output_code2
    enc_func = branch_output_code1 if transmitter_code==1 else branch_output_code2

    for _ in range(burn_in):
        input_bit = int(rng.random()<0.5)
        noiseless = enc_func(prev_state_bit, input_bit)
        r = bsc(p, noiseless, rng)
        # normalize
        r_norm = (r[0]^noiseless[0], r[1]^noiseless[1])
        state = viterbi_next_relative_metrics(state, r_norm, decoder_branch)
        prev_state_bit = input_bit

    for _ in range(steps):
        input_bit = int(rng.random()<0.5)
        noiseless = enc_func(prev_state_bit, input_bit)
        r = bsc(p, noiseless, rng)
        r_norm = (r[0]^noiseless[0], r[1]^noiseless[1])
        i = STATE_INDEX[state]
        next_state = viterbi_next_relative_metrics(state, r_norm, decoder_branch)
        j = STATE_INDEX[next_state]
        counts[i,j]+=1
        state = next_state
        prev_state_bit = input_bit

    row_sums = counts.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1
    return counts/row_sums

# Theoretical T (Eq.4) for Code-1, reordered to our state order
def theoretical_matrix_code1(p):
    q = 1-p
    T_paper = np.array([
        [q*q,    p*q,    0,        0,      p*p],
        [p*q,    0,      2*p*q,    0,      0],
        [0,      2*p*q,  0,        q*q+p*p,0],
        [0,      0,      p,        0,      0],
        [p*p,    0,      0,        0,      q*q]
    ])
    # paper’s order: (2,0),(1,0),(0,0),(0,1),(0,2)
    paper_order = [(2,0),(1,0),(0,0),(0,1),(0,2)]
    reorder_idx = [paper_order.index(s) for s in STATE_LIST]
    return T_paper[reorder_idx,:][:,reorder_idx]

# ===================================================
if __name__=="__main__":
    print("State order:", STATE_LIST)

    # 1. Random empirical Code1->Code1, p=0.1
    T_emp11 = simulate_random_empirical(P1, STEPS, BURN_IN, 1, 1, SEED)
    print("\nRandom Empirical Code1->Code1, p=0.1")
    for i,row in enumerate(T_emp11):
        print(f"{STATE_LIST[i]} -> {np.round(row,6)}")

    # 2. Random empirical Code2->Code1, p=0.1
    T_emp21 = simulate_random_empirical(P1, STEPS, BURN_IN, 2, 1, SEED)
    print("\nRandom Empirical Code2->Code1, p=0.1")
    for i,row in enumerate(T_emp21):
        print(f"{STATE_LIST[i]} -> {np.round(row,6)}")

    # 3. Random theoretical Code1->Code1, p=0.1
    T_theory01 = theoretical_matrix_code1(P1)
    print("\nRandom Theoretical Code1->Code1, p=0.1 (Eq.4)")
    for i,row in enumerate(T_theory01):
        print(f"{STATE_LIST[i]} -> {np.round(row,6)}")

    # 4. Random theoretical Code1->Code1, p=0.5
    T_theory05 = theoretical_matrix_code1(P2)
    print("\nRandom Theoretical Code1->Code1, p=0.5 (Eq.4)")
    for i,row in enumerate(T_theory05):
        print(f"{STATE_LIST[i]} -> {np.round(row,6)}")

# ---------- IMPORTS ----------
from __future__ import annotations
from typing import List, Tuple, Dict, Set
import itertools, random
import sympy as sp   # used for symbolic algebra
import csv
from pathlib import Path

# ---------- UTILITY FUNCTIONS ----------
def octal_list_to_ints(oct_list: List[int]) -> List[int]:
    return [int(str(o), 8) for o in oct_list]

def poly_degree(x: int) -> int:
    """Return the degree of a polynomial represented as an int bitmask."""
    return -1 if x == 0 else (x.bit_length() - 1)

def _mod2_div(dividend: int, divisor: int) -> Tuple[int, int]:
    """Perform polynomial long division over GF(2). Return (quotient, remainder)."""
    if divisor == 0: raise ZeroDivisionError
    q, r = 0, dividend
    d = poly_degree(divisor)
    while r and poly_degree(r) >= d:
        sh = poly_degree(r) - d
        q ^= (1 << sh)
        r ^= (divisor << sh)
    return q, r

def poly_gcd(a: int, b: int) -> int:
    """GCD of two polynomials over GF(2)."""
    A, B = a, b
    while B:
        _, r = _mod2_div(A, B)
        A, B = B, r
    return A

def polys_gcd(polys: List[int]) -> int:
    """GCD of a list of polynomials over GF(2)."""
    g = 0
    for p in polys: g = p if g == 0 else poly_gcd(g, p)
    return g

def is_delay_free(gens: List[int]) -> bool:
    """
    Check if encoder is delay-free: at least one generator polynomial
    must have a nonzero D^0 tap (i.e. LSB=1).
    """
    return any((g & 1) == 1 for g in gens)

def save_matrix_csv(states, P, filename, run_info):
    """
    Save a transition matrix to CSV, appending if file exists.
    states: list of state tuples
    P: matrix (list of lists)
    filename: csv file name
    run_info: string describing run parameters
    """
    file = Path(filename)
    with file.open("a", newline="") as f:
        writer = csv.writer(f)
        # write a run header
        writer.writerow([f"Run info: {run_info}"])
        # write header row
        writer.writerow([""] + [str(st) for st in states])
        # write each row with state label
        for st,row in zip(states, P):
            writer.writerow([str(st)] + list(row))
        writer.writerow([])  # blank line between runs

# ---------- CONVOLUTIONAL CODE TRELLIS ----------
class ConvTrellisK1:
    """
    Build the trellis for a k=1 convolutional code.

    Parameters:
        generators: list of generator polynomials (as ints).
        K: constraint length (memory+1).

    Provides:
        self.next[(state,input)] = next_state
        self.out[(state,input)]  = output n-tuple (tuple[int])
        self.preds[next_state]   = list of (state,input,output) predecessors
    """

    def __init__(self, generators: List[int], K: int):
        if K <= 0: raise ValueError("K must be >=1")
        if not generators: raise ValueError("Need generators")
        # validation checks
        for g in generators:
            if g == 0: raise ValueError("Zero generator invalid")
            if poly_degree(g) >= K: raise ValueError("Generator degree >= K")
        if not is_delay_free(generators): raise ValueError("Not delay-free")
        if polys_gcd(generators) != 1: raise ValueError("Catastrophic encoder")

        self.generators = list(generators)
        self.K = K
        self.n = len(generators)   # number of outputs per input bit
        self.m = K - 1             # memory
        self.S = 1 << max(self.m, 0)  # number of states = 2^m

        # trellis mappings
        self.next: Dict[Tuple[int,int], int] = {}
        self.out: Dict[Tuple[int,int], Tuple[int,...]] = {}
        self.preds: Dict[int, List[Tuple[int,int,Tuple[int,...]]]] = {s: [] for s in range(self.S)}

        self._build()

    def _build(self):
        """
        Build the trellis transitions by simulating all possible states (s) and inputs (u).
        """
        if self.m == 0:   # memoryless special case
            for u in (0,1):
                reg = u
                y = []
                for g in self.generators:
                    y.append((reg & 1) & (g & 1))
                ns = 0
                y = tuple(y)
                self.next[(0,u)] = ns
                self.out[(0,u)] = y
                self.preds[ns].append((0,u,y))
            return

        mask_m = (1 << self.m) - 1      # mask to keep m bits
        taps_mask = (1 << self.K) - 1   # mask to keep K taps

        # for each state s and input u, compute output and next state
        for s in range(self.S):
            for u in (0,1):
                reg = (u << self.m) | s   # shift in new bit
                y = []
                for g in self.generators:
                    taps = g & taps_mask
                    acc, t = 0, taps
                    while t:
                        lsb = t & -t
                        idx = (lsb.bit_length() - 1)
                        acc ^= (reg >> idx) & 1
                        t ^= lsb
                    y.append(acc & 1)
                ns = (reg >> 1) & mask_m
                y = tuple(y)
                self.next[(s,u)] = ns
                self.out[(s,u)] = y
                self.preds[ns].append((s,u,y))

# ---------- VITERBI METRIC UPDATE ----------
def hamming(a: Tuple[int,...], b: Tuple[int,...]) -> int:
    """Compute Hamming distance between two tuples."""
    return sum(x ^ y for x,y in zip(a,b))

def viterbi_update_relative(trellis: ConvTrellisK1, M: Tuple[int,...], r: Tuple[int,...]) -> Tuple[int,...]:
    """
    Perform one-step Viterbi metric update:
    - For each next state sp, find the best predecessor path metric.
    - Add branch Hamming distance.
    - Normalize so min(M_new)=0.
    """
    S = trellis.S
    BIG = 10**9
    M_new = [0]*S
    for sp in range(S):
        best = BIG
        for s,u,yexp in trellis.preds[sp]:
            cand = M[s] + hamming(yexp, r)
            if cand < best: best = cand
        M_new[sp] = best
    mmin = min(M_new)
    return tuple(mi - mmin for mi in M_new)

# ---------- STATE ENUMERATION ----------
def enumerate_states(trellis: ConvTrellisK1) -> List[Tuple[int,...]]:
    """
    Enumerate all reachable relative metric vectors using BFS over possible received tuples.
    """
    n = trellis.n
    all_r = [tuple(b) for b in itertools.product((0,1), repeat=n)]
    start = tuple(0 for _ in range(trellis.S))  # start at all-zero metrics
    seen: Set[Tuple[int,...]] = set([start])
    frontier = [start]
    while frontier:
        new_frontier = []
        for M in frontier:
            for r in all_r:
                M2 = viterbi_update_relative(trellis, M, r)
                if M2 not in seen:
                    seen.add(M2); new_frontier.append(M2)
        frontier = new_frontier
    return sorted(seen)

# ---------- SYMBOLIC TRANSITION MATRIX ----------
def build_transition_matrix_symbolic(trellis: ConvTrellisK1, states: List[Tuple[int,...]]):
    """
    Build transition matrix as symbolic expressions in p.
    """
    n = trellis.n
    all_r = [tuple(b) for b in itertools.product((0,1), repeat=n)]
    zero = tuple(0 for _ in range(n))
    p = sp.Symbol('p')   # symbolic variable for crossover probability

    state_index = {st:i for i,st in enumerate(states)}
    # initialize with sympy 0s
    P = [[sp.Integer(0) for _ in states] for _ in states]

    # For each state and each possible received tuple
    for i, M in enumerate(states):
        for r in all_r:
            M2 = viterbi_update_relative(trellis, M, r)
            j = state_index[M2]
            d = hamming(r, zero)
            term = (p**d)*((1-p)**(n-d))  # probability weight
            P[i][j] += term
    return P

# ---------- SIMULATION ----------
def simulate_transition_matrix(trellis: ConvTrellisK1, states: List[Tuple[int,...]],
                               p_val: float, L: int, seed: int|None=None, all_zero: bool=False):
    """
    Simulate empirical transition matrix:
    - Generate input sequence (random or all-zero).
    - Encode, send through BSC.
    - Update Viterbi metrics.
    - Count transitions between relative metric states.
    """
    rng = random.Random(seed)
    n,m = trellis.n, trellis.m

    # choose input stream
    if all_zero:
        u = [0]*L   # all-zero input
    else:
        u = [rng.randint(0,1) for _ in range(L)]  # random input
    u.extend([0]*m)  # termination

    # encode
    s = 0
    v = []
    for bit in u:
        y = trellis.out[(s,bit)]
        v.extend(y)
        s = trellis.next[(s,bit)]

    # pass through BSC
    y_recv = [b ^ int(rng.random()<p_val) for b in v]

    # group into n-tuples
    r_tuples = [tuple(y_recv[i:i+n]) for i in range(0,len(y_recv),n)]

    # track relative metrics
    state_index = {st:i for i,st in enumerate(states)}
    M = tuple(0 for _ in range(trellis.S))
    counts = [[0]*len(states) for _ in states]
    for r in r_tuples:
        M2 = viterbi_update_relative(trellis, M, r)
        i,j = state_index[M], state_index[M2]
        counts[i][j] += 1
        M = M2

    # normalize to probabilities
    P_emp = []
    for row in counts:
        s = sum(row)
        P_emp.append([c/s if s>0 else 0 for c in row])
    return P_emp

# ---------- PRETTY PRINT ----------
def print_matrix(states, P, title:str):
    """
    Print a nicely formatted table of the transition matrix with row/col labels.
    """
    state_labels = [str(st) for st in states]
    col_w = [max(len(str(P[i][j])) for i in range(len(states))) for j in range(len(states))]
    row_label_w = max(len(lbl) for lbl in state_labels)
    print(f"\n=== {title} ===")
    top = "+" + "-"*(row_label_w+2)
    for w in col_w: top += "+" + "-"*(w+2)
    top += "+"
    print(top)
    header = "| " + " "*(row_label_w) + " |"
    for lbl,w in zip(state_labels,col_w):
        header += " " + lbl.center(w) + " |"
    print(header)
    print(top)
    for st,row in zip(state_labels, P):
        label = st.ljust(row_label_w)
        line = f"| {label} |"
        for j,cell in enumerate(row):
            line += " " + str(cell).center(col_w[j]) + " |"
        print(line)
        print(top)

# ---------- MAIN ----------
def main():
    print("=== Relative Metric Markov Chain (symbolic + numeric + empirical) ===")
    n = int(input("Enter n (outputs per input): ").strip())
    m = int(input("Enter m (memory, so K=m+1): ").strip())
    gens_in = input(f"Enter {n} generators in OCTAL (space separated): ").split()
    gens = octal_list_to_ints([int(x,8) for x in gens_in])
    p_val = float(input("Enter BSC crossover probability p: ").strip())
    L = int(input("Enter length of input stream for simulation: ").strip())
    seed_in = input("Optional seed (blank for none): ").strip()
    seed = None if seed_in=="" else int(seed_in)
    all_zero_mode = input("Use all-zero input sequence for simulation? [y/N]: ").strip().lower().startswith("y")

    trellis = ConvTrellisK1(gens,m+1)
    states = enumerate_states(trellis)
    print("\n=== Reachable Relative Metric States ===")
    for st in states: print(st)

    # Symbolic matrix
    P_sym = build_transition_matrix_symbolic(trellis, states)
    print_matrix(states, P_sym, "Symbolic Transition Matrix")

    # Evaluated numeric matrix
    P_eval = [[round(float(sp.N(sp.sympify(cell).subs(sp.Symbol('p'), p_val))),3) for cell in row] for row in P_sym]
    print_matrix(states, P_eval, f"Matrix Evaluated at p={p_val}")

    # Empirical matrix
    P_emp = simulate_transition_matrix(trellis, states, p_val, L, seed, all_zero=all_zero_mode)
    P_emp_rounded = [[round(v,3) for v in row] for row in P_emp]
    sim_mode = "all-zero input" if all_zero_mode else "random input"
    print_matrix(states, P_emp_rounded, f"Empirical Transition Matrix (L={L}, {sim_mode})")
    
    run_info = f"n={n}, m={m}, gens={gens_in}, p={p_val}, L={L}, mode={'all-zero' if all_zero_mode else 'random'}"

    # Save symbolic
    save_matrix_csv(states, P_sym, "symbolic_matrix.csv", run_info)
    # Save evaluated
    save_matrix_csv(states, P_eval, "evaluated_matrix.csv", run_info)
    # Save empirical
    save_matrix_csv(states, P_emp_rounded, "empirical_matrix.csv", run_info)

    print("\nMatrices saved to CSV (symbolic_matrix.csv, evaluated_matrix.csv, empirical_matrix.csv).")


if __name__ == "__main__":
    main()

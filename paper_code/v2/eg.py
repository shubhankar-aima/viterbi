#!/usr/bin/env python3
# rel_metric_markov_symbolic.py
#
# Enumerates reachable relative metric states (k=1 conv code)
# and prints the symbolic Markov transition probability matrix (in terms of p).

from __future__ import annotations
from typing import List, Tuple, Dict, Set
import itertools

# ---------- helpers ----------
def octal_list_to_ints(oct_list: List[int]) -> List[int]:
    return [int(str(o), 8) for o in oct_list]

def poly_degree(x: int) -> int:
    return -1 if x == 0 else (x.bit_length() - 1)

def _mod2_div(dividend: int, divisor: int) -> Tuple[int, int]:
    if divisor == 0: raise ZeroDivisionError
    q, r = 0, dividend
    d = poly_degree(divisor)
    while r and poly_degree(r) >= d:
        sh = poly_degree(r) - d
        q ^= (1 << sh)
        r ^= (divisor << sh)
    return q, r

def poly_gcd(a: int, b: int) -> int:
    A, B = a, b
    while B:
        _, r = _mod2_div(A, B)
        A, B = B, r
    return A

def polys_gcd(polys: List[int]) -> int:
    g = 0
    for p in polys: g = p if g == 0 else poly_gcd(g, p)
    return g

def is_delay_free(gens: List[int]) -> bool:
    return any((g & 1) == 1 for g in gens)

# ---------- Trellis ----------
class ConvTrellisK1:
    def __init__(self, generators: List[int], K: int):
        if K <= 0: raise ValueError("K must be >=1")
        if not generators: raise ValueError("Need generators")
        for g in generators:
            if g == 0: raise ValueError("Zero generator invalid")
            if poly_degree(g) >= K: raise ValueError("Generator degree >= K")
        if not is_delay_free(generators): raise ValueError("Not delay-free")
        if polys_gcd(generators) != 1: raise ValueError("Catastrophic encoder")
        self.generators = list(generators)
        self.K = K
        self.n = len(generators)
        self.m = K - 1
        self.S = 1 << max(self.m, 0)
        self.next: Dict[Tuple[int,int], int] = {}
        self.out: Dict[Tuple[int,int], Tuple[int,...]] = {}
        self.preds: Dict[int, List[Tuple[int,int,Tuple[int,...]]]] = {s: [] for s in range(self.S)}
        self._build()

    def _build(self):
        if self.m == 0:
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

        mask_m = (1 << self.m) - 1
        taps_mask = (1 << self.K) - 1
        for s in range(self.S):
            for u in (0,1):
                reg = (u << self.m) | s
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

# ---------- Viterbi metric update ----------
def hamming(a: Tuple[int,...], b: Tuple[int,...]) -> int:
    return sum(x ^ y for x,y in zip(a,b))

def viterbi_update_relative(trellis: ConvTrellisK1, M: Tuple[int,...], r: Tuple[int,...]) -> Tuple[int,...]:
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

# ---------- Enumerator ----------
def enumerate_states(trellis: ConvTrellisK1) -> List[Tuple[int,...]]:
    n = trellis.n
    all_r = [tuple(b) for b in itertools.product((0,1), repeat=n)]
    start = tuple(0 for _ in range(trellis.S))
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

# ---------- Transition matrix (symbolic) ----------
def build_transition_matrix_symbolic(trellis: ConvTrellisK1, states: List[Tuple[int,...]]):
    n = trellis.n
    all_r = [tuple(b) for b in itertools.product((0,1), repeat=n)]
    zero = tuple(0 for _ in range(n))

    def prob_str(r):
        d = hamming(r, zero)
        if d == 0: return f"(1-p)^{n}"
        elif d == n: return f"p^{n}"
        else: return f"p^{d}(1-p)^{n-d}"

    state_index = {st:i for i,st in enumerate(states)}
    P = [[[] for _ in states] for _ in states]

    for i, M in enumerate(states):
        for r in all_r:
            M2 = viterbi_update_relative(trellis, M, r)
            j = state_index[M2]
            P[i][j].append(prob_str(r))

    P_final = []
    for row in P:
        new_row = []
        for cell in row:
            if not cell: new_row.append("0")
            else: new_row.append(" + ".join(cell))
        P_final.append(new_row)
    return P_final

# ---------- Pretty print ----------
def clean_expr(expr: str) -> str:
    expr = expr.replace("^1", "")
    expr = expr.replace("p^0", "1")
    expr = expr.replace("(1-p)^0", "1")
    expr = expr.replace("1(1-p)", "(1-p)")
    expr = expr.replace("1p", "p")
    return expr

def print_matrix(states: List[Tuple[int,...]], P: List[List[str]]):
    P_clean = [[clean_expr(cell) for cell in row] for row in P]
    col_w = [max(len(P_clean[i][j]) for i in range(len(states))) for j in range(len(states))]
    state_labels = [str(st) for st in states]
    row_label_w = max(len(lbl) for lbl in state_labels)

    print("\n=== Symbolic Transition Probability Matrix ===")
    top = "+" + "-"*(row_label_w+2)
    for w in col_w: top += "+" + "-"*(w+2)
    top += "+"
    print(top)
    header = "| " + " "*(row_label_w) + " |"
    for lbl,w in zip(state_labels,col_w):
        header += " " + lbl.center(w) + " |"
    print(header)
    print(top)

    for st,row in zip(state_labels, P_clean):
        label = st.ljust(row_label_w)
        line = f"| {label} |"
        for j,cell in enumerate(row):
            line += " " + cell.center(col_w[j]) + " |"
        print(line)
        print(top)


# ---------- Main ----------
def main():
    print("=== Relative Metric Markov Chain (symbolic) ===")
    n = int(input("Enter n (outputs per input): ").strip())
    m = int(input("Enter m (memory, so K=m+1): ").strip())
    K = m+1
    gens_in = input(f"Enter {n} generators in OCTAL (space separated): ").split()
    gens = octal_list_to_ints([int(x,8) for x in gens_in])

    trellis = ConvTrellisK1(gens,K)
    states = enumerate_states(trellis)

    print("\n=== Reachable Relative Metric States ===")
    for idx, st in enumerate(states):
        print(f"{idx}: {st}")

    P = build_transition_matrix_symbolic(trellis, states)
    print_matrix(states, P)

if __name__ == "__main__":
    main()

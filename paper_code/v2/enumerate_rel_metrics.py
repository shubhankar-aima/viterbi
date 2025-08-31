# k=1 (rate 1/n) feedforward conv code; random input; BSC(p); Viterbi metric vectors.

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional, Set
import itertools, random

# ----------------- GF(2) helpers -----------------
def octal_list_to_ints(oct_list: List[int]) -> List[int]:
    return [int(str(o), 8) for o in oct_list]

def poly_degree(x: int) -> int:
    if x == 0: return -1
    return x.bit_length() - 1

def _mod2_div(dividend: int, divisor: int) -> Tuple[int, int]:
    if divisor == 0: raise ZeroDivisionError("Division by zero polynomial.")
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
    for p in polys:
        g = p if g == 0 else poly_gcd(g, p)
    return g

def is_delay_free(gens: List[int]) -> bool:
    return any((g & 1) == 1 for g in gens)  # D^0 tap exists in at least one generator

# --------------- Trellis (k=1) -------------------
class ConvTrellisK1:
    """
    k=1, rate 1/n feedforward conv encoder trellis, K=m+1
    Provides:
      next_state[(s,u)] -> s'
      out[(s,u)] -> expected n-tuple (tuple[int])
      preds[s'] -> list of (s,u,out) predecessors
    """
    def __init__(self, generators: List[int], K: int):
        if K <= 0: raise ValueError("K must be >= 1")
        if not generators: raise ValueError("Provide at least one generator.")
        for g in generators:
            if g == 0: raise ValueError("Zero generator invalid.")
            if poly_degree(g) >= K: raise ValueError(f"Generator {bin(g)} degree >= K={K}.")
        if not is_delay_free(generators):
            raise ValueError("Not delay-free: at least one generator must have D^0 tap.")
        if polys_gcd(generators) != 1:
            raise ValueError("Catastrophic encoder: generators share common factor.")

        self.generators = list(generators)
        self.K = int(K)
        self.n = len(generators)
        self.m = K - 1
        self.S = 1 << max(self.m, 0)

        self.next_state: Dict[Tuple[int,int], int] = {}
        self.out:        Dict[Tuple[int,int], Tuple[int,...]] = {}
        self.preds:      Dict[int, List[Tuple[int,int,Tuple[int,...]]]] = {s: [] for s in range(self.S)}
        self._build()

    def _build(self) -> None:
        if self.m == 0:
            for u in (0,1):
                reg = u
                y = []
                for g in self.generators:
                    y.append((reg & 1) & (g & 1))
                ns = 0
                y = tuple(y)
                self.next_state[(0,u)] = ns
                self.out[(0,u)] = y
                self.preds[ns].append((0,u,y))
            return

        mask_m = (1 << self.m) - 1
        taps_mask = (1 << self.K) - 1
        for s in range(self.S):
            for u in (0,1):
                reg = (u << self.m) | s
                # expected n-tuple
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
                self.next_state[(s,u)] = ns
                self.out[(s,u)] = y
                self.preds[ns].append((s,u,y))

# --------------- Encoder (k=1, zero-terminated) ---------------
class ConvEncoderZero:
    def __init__(self, generators: List[int], K: int):
        self.trellis = ConvTrellisK1(generators, K)
        self.m = self.trellis.m
        self.n = self.trellis.n
        self.S = self.trellis.S

    def encode(self, u_bits: List[int]) -> List[int]:
        s = 0
        out: List[int] = []
        # payload
        for u in u_bits:
            y = self.trellis.out[(s,u)]
            out.extend(y)
            s = self.trellis.next_state[(s,u)]
        # zero termination (append m zeros)
        if self.m > 0:
            for _ in range(self.m):
                y = self.trellis.out[(s,0)]
                out.extend(y)
                s = self.trellis.next_state[(s,0)]
            if s != 0:
                raise RuntimeError("Zero termination failed.")
        return out

# --------------- BSC -----------------
class BSC:
    def __init__(self, p: float, seed: Optional[int] = None):
        if not (0.0 <= p <= 1.0): raise ValueError("p must be in [0,1]")
        self.p = float(p)
        self.rng = random.Random(seed)

    def transmit(self, bits: Iterable[int]) -> Tuple[List[int], List[int]]:
        y, flips = [], []
        for i,b in enumerate(bits):
            if b not in (0,1): raise ValueError("Bits must be 0/1.")
            f = self.rng.random() < self.p
            yb = b ^ int(f)
            y.append(yb)
            if f: flips.append(i)
        return y, flips

# --------------- Viterbi metric updates -----------------
def hamming(a: Tuple[int,...], b: Tuple[int,...]) -> int:
    return sum(x ^ y for x,y in zip(a,b))

def viterbi_update_relative(
    trellis: ConvTrellisK1,
    M_rel: Tuple[int, ...],
    r: Tuple[int, ...],
) -> Tuple[int, ...]:
    S = trellis.S
    M_new = [0]*S
    BIG = 10**9
    for sp in range(S):
        best = BIG
        for s,u,yexp in trellis.preds[sp]:
            cand = M_rel[s] + hamming(yexp, r)
            if cand < best: best = cand
        M_new[sp] = best
    mmin = min(M_new)
    return tuple(mi - mmin for mi in M_new)

# --------------- CLI helpers -----------------
def prompt_int(prompt: str, min_val: Optional[int]=None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"Please enter an integer >= {min_val}."); continue
            return v
        except ValueError:
            print("Please enter an integer.")

def prompt_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if not (lo <= v <= hi):
                print(f"Please enter a value in [{lo},{hi}]."); continue
            return v
        except ValueError:
            print("Please enter a number.")

# --------------- Main ---------------
def main():
    print("=== Relative Metric Enumerator / Random Run ===")
    print("k=1 (rate=1/n). Zero-termination is enforced for random run.")
    mode = input("Choose mode: [E]numerator (all states) or [R]andom run? ").strip().lower()

    n = prompt_int("Enter n (outputs per input, number of generators): ", 1)
    m = prompt_int("Enter m (memory, so K = m + 1): ", 0)
    K = m + 1

    print(f"Enter {n} generator polynomials in OCTAL (e.g., for [111,101] enter: 7 5).")
    while True:
        parts = input(f"Generators (octal) [{n} numbers, space-separated]: ").strip().split()
        try:
            if len(parts) != n: raise ValueError(f"Need exactly {n} octal numbers.")
            octals = [int(x,8) for x in parts]
            gens = octal_list_to_ints(octals)
            break
        except Exception as e:
            print(f"Invalid entry: {e}")

    trellis = ConvTrellisK1(gens, K)

    if mode.startswith("e"):   # Enumerator mode
        all_r = [tuple(bits) for bits in itertools.product((0,1), repeat=n)]
        start = tuple(0 for _ in range(trellis.S))
        seen: Set[Tuple[int,...]] = set([start])
        frontier = [start]

        while frontier:
            new_frontier = []
            for M in frontier:
                for r in all_r:
                    M2 = viterbi_update_relative(trellis, M, r)
                    if M2 not in seen:
                        seen.add(M2)
                        new_frontier.append(M2)
            frontier = new_frontier

        print("\n=== UNIQUE RELATIVE METRICS (Enumerator mode) ===")
        for st in sorted(seen):
            print(st)

    else:   # Random run mode
        L = prompt_int("Enter payload length L (random input bits): ", 1)
        seed_s = input("Optional RNG seed for input (blank for none): ").strip()
        seed = None if seed_s == "" else int(seed_s)
        rnd = random.Random(seed)
        u = [rnd.randint(0,1) for _ in range(L)]

        p = prompt_float("Enter BSC crossover probability p in [0,1]: ", 0.0, 1.0)
        bsc_seed_s = input("Optional BSC RNG seed (blank for none): ").strip()
        bsc_seed = None if bsc_seed_s == "" else int(bsc_seed)

        # Encode and transmit
        enc = ConvEncoderZero(gens, K)
        v = enc.encode(u)
        ch = BSC(p, bsc_seed)
        y, flips = ch.transmit(v)

        # Slice into n-tuples
        r_tuples = [tuple(y[i:i+n]) for i in range(0,len(y),n)]

        # Viterbi trajectory
        M = tuple(0 for _ in range(trellis.S))
        seen: Set[Tuple[int,...]] = set([M])
        for r in r_tuples:
            M = viterbi_update_relative(trellis, M, r)
            seen.add(M)

        print("\n=== UNIQUE RELATIVE METRICS (Random run mode) ===")
        for st in sorted(seen):
            print(st)

if __name__ == "__main__":
    main()

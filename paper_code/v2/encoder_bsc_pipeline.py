#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional
import random

# =========================
# GF(2) polynomial helpers
# =========================
def octal_list_to_ints(oct_list: List[int]) -> List[int]:
    """[7,5] (octal) -> [0b111,0b101] (ints); LSB = D^0."""
    return [int(str(o), 8) for o in oct_list]

def poly_degree(x: int) -> int:
    if x == 0:
        return -1
    return x.bit_length() - 1

def _mod2_div(dividend: int, divisor: int) -> Tuple[int, int]:
    if divisor == 0:
        raise ZeroDivisionError("Division by zero polynomial.")
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
    # At least one generator has D^0 tap → odd integer
    return any((g & 1) == 1 for g in gens)

# ==============================================
# Feedforward conv. encoder (k=1, rate 1/n)
#   - ALWAYS zero-terminated (append m zeros)
#   - Optional verbose logging
# ==============================================
class _Printer:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
    def __call__(self, *args, **kwargs):
        if self.enabled:
            print(*args, **kwargs)

class ConvEncoderZero:
    """
    Binary convolutional encoder (k=1, rate 1/n), feedforward only.
    ALWAYS zero-terminates: appends m = K-1 zeros so the codeword ends in state 0.

    Args:
        generators: list of FIR polynomials as ints (LSB=D^0), e.g. [0b111, 0b101]
        K: constraint length (m = K-1); max deg(g_i) < K
        verbose: if True, prints intermediary steps (trellis build, encoding steps, termination)
    """
    def __init__(self, generators: List[int], K: int, verbose: bool = False):
        self.generators = list(generators)
        self.K = int(K)
        self.m = self.K - 1
        self.n = len(self.generators)
        self._print = _Printer(verbose)

        # ---- validation: feedforward + delay-free + non-catastrophic ----
        if self.K <= 0:
            raise ValueError("K must be >= 1.")
        if self.n <= 0:
            raise ValueError("Provide at least one generator (rate 1/n).")
        for g in self.generators:
            if g == 0:
                raise ValueError("Zero generator is invalid.")
            if poly_degree(g) >= self.K:
                raise ValueError(f"Generator {bin(g)} degree >= K={self.K}.")
        if not is_delay_free(self.generators):
            raise ValueError("Not delay-free: at least one generator must have a nonzero D^0 tap.")
        if polys_gcd(self.generators) != 1:
            raise ValueError("Catastrophic encoder: generators share a nontrivial common factor over GF(2).")

        # ---- trellis ----
        self.num_states = 1 << max(self.m, 0)
        self.trellis_next: Dict[Tuple[int, int], int] = {}
        self.trellis_out:  Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self._build_trellis()

    def _state_bits(self, s: int) -> str:
        if self.m == 0:
            return "∅"
        return format(s, f"0{self.m}b")

    def _build_trellis(self) -> None:
        if self.m == 0:
            # memoryless
            for u in (0, 1):
                reg = u
                y = []
                for g in self.generators:
                    y.append((reg & 1) & (g & 1))
                self.trellis_next[(0, u)] = 0
                self.trellis_out[(0, u)]  = tuple(y)
            return

        mask_m = (1 << self.m) - 1
        for s in range(self.num_states):
            for u in (0, 1):
                reg = (u << self.m) | s
                # outputs
                y = []
                for g in self.generators:
                    taps = g & ((1 << self.K) - 1)
                    acc, t = 0, taps
                    while t:
                        lsb = t & -t
                        idx = (lsb.bit_length() - 1)
                        acc ^= (reg >> idx) & 1
                        t ^= lsb
                    y.append(acc & 1)
                # next state
                ns = (reg >> 1) & mask_m
                self.trellis_next[(s, u)] = ns
                self.trellis_out[(s, u)]  = tuple(y)

    # Always zero-terminated encode
    def encode(self, u_bits: List[int], show_steps: bool = True) -> List[int]:
        if any(b not in (0, 1) for b in u_bits):
            raise ValueError("Input bits must be 0/1.")
        coded, s = self._encode_from_state(u_bits, start_state=0, show_steps=show_steps, tag="PAYLD")
        if self.m > 0:
            tail = [0] * self.m
            tail_coded, end2 = self._encode_from_state(tail, start_state=s, show_steps=show_steps, tag="TAIL ")
            coded.extend(tail_coded)
            if end2 != 0:
                raise RuntimeError("Zero termination failed to end in state 0.")
        return coded

    def _encode_from_state(
        self,
        u_bits: Iterable[int],
        start_state: int,
        show_steps: bool = True,
        tag: str = ""
    ) -> Tuple[List[int], int]:
        if not (0 <= start_state < self.num_states):
            raise ValueError("start_state out of range.")
        s = start_state
        out: List[int] = []

        if show_steps and self._print.enabled:
            print(f"{'#':>3} {'tag':<5} {'state':>6} {'u':>1} {'y':<10} {'next':>6}")
            print("-" * 38)

        step = 0
        for u in u_bits:
            y = self.trellis_out[(s, u)]
            ns = self.trellis_next[(s, u)]
            out.extend(y)
            if show_steps and self._print.enabled:
                print(f"{step:>3} {tag:<5} {s:>6}({self._state_bits(s)}) {u:>1} {str(y):<10} {ns:>6}({self._state_bits(ns)})")
            s = ns
            step += 1
        return out, s

# -------------
# Binary Symmetric Channel
# -------------
class BSC:
    """Binary Symmetric Channel with crossover probability p."""
    def __init__(self, p: float, seed: Optional[int] = None, verbose: bool = False):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1].")
        self.p = float(p)
        self.verbose = verbose
        self.random = random.Random(seed)

    def transmit(self, bits: Iterable[int]) -> Tuple[List[int], List[int]]:
        y: List[int] = []
        flips: List[int] = []
        for i, b in enumerate(bits):
            if b not in (0, 1):
                raise ValueError("Bits must be 0/1.")
            flip = self.random.random() < self.p
            out = b ^ int(flip)
            y.append(out)
            if flip:
                flips.append(i)
        if self.verbose:
            self._print_report(bits, y, flips)
        return y, flips

    def _print_report(self, x: Iterable[int], y: Iterable[int], flips: Iterable[int]) -> None:
        x = list(x); y = list(y); flips = list(flips)
        print("\n=== BSC REPORT ===")
        print(f"p = {self.p}")
        print(f"n = {len(x)} bits")
        print("x (in): ", x)
        print("y (out):", y)
        print("flip idx:", flips)
        if flips:
            print(f"Total flips: {len(flips)} ({len(flips)/len(x):.2%})")
        else:
            print("Total flips: 0 (0.00%)")
        print("=" * 32)

# -------------------------------
# CLI Glue
# -------------------------------
def prompt_int(prompt: str, min_val: Optional[int] = None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"Please enter an integer >= {min_val}.")
                continue
            return v
        except ValueError:
            print("Please enter an integer.")

def prompt_float(prompt: str, lo: float, hi: float) -> float:
    while True:
        s = input(prompt).strip()
        try:
            v = float(s)
            if not (lo <= v <= hi):
                print(f"Please enter a value in [{lo},{hi}].")
                continue
            return v
        except ValueError:
            print("Please enter a number.")

def main():
    print("=== Convolutional Encoder + BSC Pipeline ===")
    print("k is fixed at 1 (rate = 1/n). You will enter n and m, then the n generators in OCTAL.")
    n = prompt_int("Enter n (outputs per input, number of generators): ", min_val=1)
    m = prompt_int("Enter m (memory, so K = m + 1): ", min_val=0)
    K = m + 1

    print(f"Enter {n} generator polynomials in OCTAL (e.g., for [111,101] enter: 7 5).")
    while True:
        parts = input(f"Generators (octal) [{n} numbers, space-separated]: ").strip().split()
        try:
            if len(parts) != n:
                raise ValueError(f"Need exactly {n} octal numbers.")
            octals = [int(x, 8) for x in parts]  # accept '7' as octal
            gens = octal_list_to_ints(octals)
            break
        except Exception as e:
            print(f"Invalid entry: {e}")

    L = prompt_int("Enter payload length L (random input bits): ", min_val=1)
    seed_s = input("Optional RNG seed (blank for none): ").strip()
    seed = None if seed_s == "" else int(seed_s)

    verbose_enc = input("Verbose encoder steps? [y/N]: ").strip().lower().startswith("y")

    p = prompt_float("Enter BSC crossover probability p in [0,1]: ", 0.0, 1.0)
    bsc_seed_s = input("Optional BSC RNG seed (blank for none): ").strip()
    bsc_seed = None if bsc_seed_s == "" else int(bsc_seed_s)
    verbose_bsc = input("Verbose BSC report? [y/N]: ").strip().lower().startswith("y")

    # Generate random input payload
    rnd = random.Random(seed)
    u = [rnd.randint(0, 1) for _ in range(L)]

    # Build encoder and encode (always zero-terminated)
    enc = ConvEncoderZero(generators=gens, K=K, verbose=verbose_enc)
    v = enc.encode(u, show_steps=verbose_enc)

    # Send through BSC
    ch = BSC(p=p, seed=bsc_seed, verbose=verbose_bsc)
    y, flips = ch.transmit(v)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"n = {n}, m = {m}, K = {K}")
    print(f"Generators (bin): {[bin(g) for g in gens]}")
    print(f"Payload length L: {L} (zero-termination adds {m} input zeros)")
    print(f"Codeword length:  {len(v)} = n * (L + m) = {n} * ({L} + {m})")
    print(f"BSC p: {p}")
    print("\nInput u (random):")
    print(u)
    print("\nEncoded v:")
    print(v)
    print("\nReceived y (after BSC):")
    print(y)
    print("\nFlip indices:", flips)
    print("=== DONE ===")

if __name__ == "__main__":
    main()

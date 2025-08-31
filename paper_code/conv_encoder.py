
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable, Optional

# =========================
# GF(2) polynomial helpers
# =========================

def octal_list_to_ints(oct_list: List[int]) -> List[int]:
    """[0o7,0o5] -> [0b111,0b101]; LSB = D^0."""
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
#   - ALWAYS zero-terminated
#   - Verbose logging of every intermediary step
# ==============================================

class _Printer:
    """Lightweight optional printer to keep the core logic clean."""
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
        generators: list of FIR polynomials as ints (LSB=D^0), e.g. [0b1, 0b11]
        K: constraint length (m = K-1); max deg(g_i) < K
        verbose: if True, prints intermediary steps (trellis build, encoding steps, termination)

    Properties:
        n, m, num_states
        trellis_next[(s,u)] -> next_state
        trellis_out[(s,u)]  -> tuple of n output bits
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

        self._print(">> INIT")
        self._print(f"   K={self.K}, m={self.m}, n={self.n}, generators={self._gens_as_str()}")

        # ---- trellis ----
        self.num_states = 1 << max(self.m, 0)
        self.trellis_next: Dict[Tuple[int, int], int] = {}
        self.trellis_out:  Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self._build_trellis()

    # Human-friendly formatting helpers
    def _gens_as_str(self) -> str:
        return "[" + ", ".join(f"{g:#b}" for g in self.generators) + "]"

    def _state_bits(self, s: int) -> str:
        if self.m == 0:
            return "∅"
        return format(s, f"0{self.m}b")

    def _build_trellis(self) -> None:
        self._print(">> BUILD TRELLIS")
        if self.m == 0:
            # memoryless
            for u in (0, 1):
                reg = u
                y = []
                for g in self.generators:
                    y.append((reg & 1) & (g & 1))
                self.trellis_next[(0, u)] = 0
                self.trellis_out[(0, u)]  = tuple(y)
                self._print(f"   s=0,u={u} -> s'={0}, y={tuple(y)}")
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
                self._print(f"   s={s:>2}({self._state_bits(s)}), u={u} -> s'={ns:>2}({self._state_bits(ns)}), y={tuple(y)}")

    # ---------------------------------
    # Public API (always zero-terminated)
    # ---------------------------------

    def encode(self, u_bits: List[int], show_steps: bool = True) -> List[int]:
        """
        Encode and then append m zeros to force end-state = 0.
        Output length = n * (len(u_bits) + m).

        If show_steps is True, prints a table of each step (state, input, outputs, next state).
        """
        if any(b not in (0, 1) for b in u_bits):
            raise ValueError("Input bits must be 0/1.")
        self._print(">> ENCODE (payload)")
        coded, s = self._encode_from_state(u_bits, start_state=0, show_steps=show_steps, tag="PAYLOAD")
        if self.m > 0:
            tail = [0] * self.m
            self._print(f">> ZERO TERMINATION (append {self.m} zeros)")
            tail_coded, end2 = self._encode_from_state(tail, start_state=s, show_steps=show_steps, tag="TAIL   ")
            coded.extend(tail_coded)
            if end2 != 0:
                raise RuntimeError("Zero termination failed to end in state 0.")
            self._print(">> END STATE CONFIRMED: 0")
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
            print(f"   {'#':>3}  {'tag':<7}  {'state':>6}  {'u':>1}  {'y (n-tuple)':<12}  {'next':>6}")
            print("   " + "-" * 44)

        step = 0
        for u in u_bits:
            y = self.trellis_out[(s, u)]
            ns = self.trellis_next[(s, u)]
            out.extend(y)
            if show_steps and self._print.enabled:
                print(f"   {step:>3}  {tag:<7}  {s:>6}({self._state_bits(s)})  {u:>1}   {str(y):<12}  {ns:>6}({self._state_bits(ns)})")
            s = ns
            step += 1
        return out, s

    def trellis_tables(self) -> Tuple[List[List[int]], List[List[Tuple[int, ...]]]]:
        NEXT = [[0, 0] for _ in range(self.num_states)]
        OUT  = [[(0,) * self.n, (0,) * self.n] for _ in range(self.num_states)]
        for s in range(self.num_states):
            for u in (0, 1):
                NEXT[s][u] = self.trellis_next[(s, u)]
                OUT[s][u]  = self.trellis_out[(s, u)]
        return NEXT, OUT

    # -------- helpers for Section 3 (branch metrics) --------

    def branch_output(self, state: int, u: int) -> Tuple[int, ...]:
        return self.trellis_out[(state, u)]

    def branch_weight(self, state: int, u: int, ref: Tuple[int, ...]) -> int:
        y = self.trellis_out[(state, u)]
        if len(y) != len(ref):
            raise ValueError("Reference tuple length mismatch.")
        return sum((a ^ b) & 1 for a, b in zip(y, ref))


# -------------------------------
# Convenience: from octal gens
# -------------------------------

def build_encoder_from_octal(oct_gens: List[int], K: int, verbose: bool = False) -> ConvEncoderZero:
    return ConvEncoderZero(octal_list_to_ints(oct_gens), K, verbose=verbose)


# -------------------------------
# Pretty printers
# -------------------------------

def print_trellis(encoder: ConvEncoderZero) -> None:
    print("=== TRELLIS ===")
    for s in range(encoder.num_states):
        for u in (0, 1):
            ns = encoder.trellis_next[(s, u)]
            y  = encoder.trellis_out[(s, u)]
            print(f"s={s:>2}({encoder._state_bits(s)}), u={u} -> s'={ns:>2}({encoder._state_bits(ns)}), y={y}")
    print("=" * 32)


def print_generator_info(encoder: ConvEncoderZero) -> None:
    print("=== GENERATORS ===")
    for i, g in enumerate(encoder.generators):
        taps = [idx for idx in range(encoder.K) if (g >> idx) & 1]
        print(f"g{i}: {g:#b} (taps D^{{{','.join(map(str, taps))}}})")
    print("=" * 32)


if __name__ == "__main__":
    # Quick demo
    enc = build_encoder_from_octal([0o7, 0o5], K=3, verbose=True)  # gens [111,101], K=3 => m=2
    print_generator_info(enc)
    print_trellis(enc)
    u = [1, 0, 1, 1]
    v = enc.encode(u, show_steps=True)
    print("Input u:", u)
    print("Coded v:", v)

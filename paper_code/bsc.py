
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable
import random

class BSC:
    """Binary Symmetric Channel with crossover probability p.
    Provides detailed logs of which bit positions flipped.
    """
    def __init__(self, p: float, seed: Optional[int] = None, verbose: bool = False):
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0,1].")
        self.p = float(p)
        self.verbose = verbose
        self.random = random.Random(seed)

    def transmit(self, bits: Iterable[int]) -> Tuple[List[int], List[int]]:
        """Transmit bits through BSC.
        Returns:
            y: received bits (after potential flips)
            flips: list of indices where a flip occurred
        """
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
        print("=== BSC REPORT ===")
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


if __name__ == "__main__":
    x = [1,0,1,1,0,0,1,0,1,0]
    ch = BSC(p=0.3, seed=42, verbose=True)
    y, flips = ch.transmit(x)

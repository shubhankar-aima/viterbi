import random
import csv
from viterbi import *
from bindata import BinData
from tqdm import tqdm

# Convolutional code generator polynomials (rate 1/2 code example)
GEN_POLYS = [0b111, 0b101]
MSG_LEN = 100           # bits per trial
NUM_TRIALS = 1000       # trials per p
P_VALUES = [i / 1000 for i in range(1, 499)]  # p from 0.001 to 0.1 in steps of 0.001

CSV_FILENAME = "ber_vs_p_results.csv"

def flip_bits_bitstring(bitstring, p):
    """Flip bits inside a BinData with probability p."""
    bitlist = list(str(bitstring))
    bitlist_int = [int(b) for b in bitlist]

    for i in range(len(bitlist_int)):
        if random.random() < p:
            bitlist_int[i] = 1 - bitlist_int[i]

    flipped_bitstring = ''.join(str(b) for b in bitlist_int)
    return BinData(flipped_bitstring)

def simulate_ber_for_p(p, transitions):
    total_errors = 0
    total_bits = 0
    for _ in tqdm(range(NUM_TRIALS), desc=f"Simulating p={p:.3f}", unit="trial", leave=False):
        msg_bits = ''.join(str(random.randint(0, 1)) for _ in range(MSG_LEN))
        msg = BinData(msg_bits)

        encoded = transitions.encode(msg)
        noisy = flip_bits_bitstring(encoded, p)
        decoded = transitions.decode(noisy)

        errors = sum(m != d for m, d in zip(str(msg), str(decoded)))
        total_errors += errors
        total_bits += MSG_LEN

    ber = total_errors / total_bits
    return ber

def main():
    transitions = Transitions(GEN_POLYS)

    # Prepare CSV file with headers
    with open(CSV_FILENAME, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Crossover Probability (p)", "Empirical BER"])

    for p in P_VALUES:
        ber = simulate_ber_for_p(p, transitions)

        # Append result to CSV
        with open(CSV_FILENAME, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([f"{p:.3f}", f"{ber:.8f}"])

        # Print intermediate result for monitoring
        print(f"p = {p:.3f}, Empirical BER = {ber:.8f}")

    print(f"\nAll simulations done. Results saved to {CSV_FILENAME}")

if __name__ == "__main__":
    main()

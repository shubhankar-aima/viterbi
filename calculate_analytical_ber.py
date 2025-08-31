import csv

CSV_FILENAME = "ber_vs_p_results.csv"
OUTPUT_CSV_FILENAME = "ber_vs_p_results_with_tiebreak.csv"

def pe_lexicographic(p):
    # Implements formula (11) from your uploaded image
    numerator = 7*p**2 - 12*p**3 + 10*p**4 - 4*p**5
    denominator = 1 + 3*p**2 - 2*p**3
    return numerator / denominator if denominator != 0 else 0.0

def pe_antilexicographic(p):
    # Implements formula (12) from your uploaded image
    numerator = p**2 * (7 - 8*p - 8*p**2 + 26*p**3 - 24*p**4 + 8*p**5)
    denominator = 1 + 3*p**2 - 2*p**3
    return numerator / denominator if denominator != 0 else 0.0

def pe_coinflip(p):
    # Implements formula (16) from your uploaded image
    numerator = p**2 * (14 - 23*p + 16*p**2 + 2*p**3 - 16*p**4 + 8*p**5)
    denominator = (1 + 3*p**2 - 2*p**3) * (2 - p + 4*p**2 - 4*p**3)
    return numerator / denominator if denominator != 0 else 0.0

def main():
    # Read simulation data
    p_values = []
    sim_ber_values = []
    with open(CSV_FILENAME, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            p_values.append(float(row["Crossover Probability (p)"]))
            sim_ber_values.append(float(row["Empirical BER"]))

    # Calculate exact P_e for each tie-breaking rule
    lex_pe = []
    anti_lex_pe = []
    coinflip_pe = []
    for p in p_values:
        lex_pe.append(pe_lexicographic(p))
        anti_lex_pe.append(pe_antilexicographic(p))
        coinflip_pe.append(pe_coinflip(p))

    # Write all results to new CSV
    with open(OUTPUT_CSV_FILENAME, mode='w', newline='') as csvfile:
        fieldnames = [
            "Crossover Probability (p)",
            "Empirical BER",
            "P_e Lexicographic",
            "P_e Anti-Lexicographic",
            "P_e Coin-Flip"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p, sim_ber, lp, alp, cfp in zip(p_values, sim_ber_values, lex_pe, anti_lex_pe, coinflip_pe):
            writer.writerow({
                "Crossover Probability (p)": f"{p:.3f}",
                "Empirical BER": f"{sim_ber:.8f}",
                "P_e Lexicographic": f"{lp:.8f}",
                "P_e Anti-Lexicographic": f"{alp:.8f}",
                "P_e Coin-Flip": f"{cfp:.8f}",
            })

    print(f"Exact $P_e$ values for all tie-breakers saved to {OUTPUT_CSV_FILENAME}")

if __name__ == "__main__":
    main()

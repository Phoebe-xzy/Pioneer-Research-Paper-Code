"""
This script is all about solving Pell equations of the form x^2 - N y^2 = 1 using the Chakravala method.

What this script does:
- We pick non-square numbers N from a range [low, high], making sure to get a good spread across different sizes by cutting the log10 scale into bins.
- For each chosen N, we run the Chakravala method to find the smallest solution (x, y).
- Finally, we save all the results to an Excel file, storing big numbers as strings so they don’t get messed up in spreadsheets.

You can run it from the command line with options for the range, how many samples per bin, random seed, and output file.
"""

import pandas as pd
import math
from tqdm import tqdm
import random
from typing import Optional
import argparse
from datetime import datetime

# Helper: Efficiently sample k distinct non-square integers in [low, high]
def sample_non_squares(low: int, high: int, k: int, seed: Optional[int] = None):
    """
    Takes low, high, and how many numbers you want. Makes sure they aren’t squares. Gives back a list of sampled N.
    """
    if seed is not None:
        random.seed(seed)
    chosen = set()
    while len(chosen) < k:
        n = random.randint(low, high)
        r = math.isqrt(n)
        if r * r == n:  # skip perfect squares
            continue
        chosen.add(n)
    return sorted(chosen)

# Stratified sampling on log10(N): fixed number per bin
def stratified_sample_non_squares(low: int, high: int, num_bins: int = 6, per_bin: int = 20, seed: Optional[int] = None):
    """
    We cut the log10(N) range into bins, then randomly grab numbers from each. That way we cover small and large N fairly.
    Takes the range, number of bins, and how many to pick per bin, then returns a list of non-square N spread across those bins.
    """
    if seed is not None:
        random.seed(seed)
    # Build log10 edges
    lo_log = math.log10(max(2, low))
    hi_log = math.log10(high)
    edges_log = [lo_log + i * (hi_log - lo_log) / num_bins for i in range(num_bins + 1)]

    chosen = []
    for i in range(num_bins):
        # Convert log edges back to integer bounds for this bin
        L = int(max(low, math.ceil(10 ** edges_log[i])))
        R = int(min(high, math.floor(10 ** edges_log[i + 1] - 1)))
        if L > R:
            # This bin doesn't have any integers, so skip it
            continue
        S = set()
        attempts = 0
        max_attempts = per_bin * 200
        while len(S) < per_bin and attempts < max_attempts:
            n = random.randint(L, R)
            r = math.isqrt(n)
            if r * r != n:  # keep only non-squares
                S.add(n)
            attempts += 1
        chosen.extend(sorted(S))
    return sorted(chosen)

def chakravala_method(N: int) -> tuple[int, int]:
    """Find the smallest (x, y) solving x^2 - N y^2 = 1 using the Chakravala method.

    If N is a square, we stop. If something goes wrong with divisibility, we also stop.
    """
    # Check if N is a perfect square
    sqrt_N = int(math.isqrt(N))
    if sqrt_N * sqrt_N == N:
        raise ValueError("N cannot be a perfect square.")

    # Start with a, b, k chosen so that k is small and positive, using ceil(sqrt(N)) to get started
    a = sqrt_N + 1
    b = 1
    k = a * a - N * b * b


    # Keep going until k becomes 1, which means we found the solution
    while k != 1:
        # Pick an m so that (a + b*m) divides nicely by k and also keeps m^2 close to N
        m_candidates = []
        limit = int(math.isqrt(N)) + 1
        # We look at non-negative m values, avoiding the case where a + b*m = 0 to keep b nonzero
        for m in range(0, limit + 1):
            if (a + b * m) % k == 0 and (a + b * m) != 0:
                m_candidates.append(m)

        # If no candidates found, widen the search range until we find some
        while not m_candidates:
            limit *= 2
            for m in range(0, limit + 1):
                if (a + b * m) % k == 0 and (a + b * m) != 0:
                    m_candidates.append(m)

        # Pick the m that keeps m^2 closest to N
        m = min(m_candidates, key=lambda m: abs(m * m - N))

        # Update a, b, k using the Brahmagupta-Fibonacci identity (the heart of Chakravala)
        # Make sure the divisions work out to integers here
        if (a * m + N * b) % abs(k) != 0 or (a + b * m) % abs(k) != 0:
            raise ValueError("Non-integer solution encountered during iteration.")
        a_new = (a * m + N * b) // abs(k)
        b_new = (a + b * m) // abs(k)
        k_new = (m * m - N) // k

        a, b, k = a_new, b_new, k_new

    # Double-check that the solution is correct
    if a * a - N * b * b != 1:
        raise ValueError(f"Final result does not satisfy Pell equation: {a}^2 - {N}*{b}^2 != 1")
    return a, b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample non-square N and compute minimal Pell solutions via Chakravala.")
    parser.add_argument("--low", type=int, default=2, help="Lower bound for N (inclusive, >= 2)")
    parser.add_argument("--high", type=int, default=10_000_000, help="Upper bound for N (inclusive)")
    parser.add_argument("--bins", type=int, default=6, help="Number of equal-width bins on log10(N) for stratified sampling")
    parser.add_argument("--per-bin", type=int, default=30, dest="per_bin", help="Target number of samples per bin")
    parser.add_argument("--seed", type=int, default=20250818, help="Random seed for reproducibility")
    parser.add_argument("--out", type=str, default=None, help="Output Excel file path (.xlsx). If omitted, a descriptive name is generated")

    args = parser.parse_args()

    if args.low < 2:
        parser.error("--low must be >= 2")
    if args.high <= args.low:
        parser.error("--high must be greater than --low")
    if args.bins <= 0:
        parser.error("--bins must be positive")
    if args.per_bin <= 0:
        parser.error("--per-bin must be positive")

    # Stratified sampling parameters
    num_bins = args.bins
    per_bin = args.per_bin

    # Sample N
    sampled_N = stratified_sample_non_squares(args.low, args.high, num_bins=num_bins, per_bin=per_bin, seed=args.seed)

    # Coverage report
    try:
        s = pd.Series(sampled_N, name="N")
        s_log = s.apply(lambda v: math.log10(v))
        edges_log = [s_log.min() + i * (s_log.max() - s_log.min()) / num_bins for i in range(num_bins + 1)]
        cats = pd.cut(s_log, bins=edges_log, include_lowest=True)
        counts = cats.value_counts().sort_index()
        print("[Stratified sampling] Bin counts (log10 N bins):")
        for iv, c in counts.items():
            print(f"  {iv}: n = {int(c)}")
    except Exception as e:
        print(f"[Stratified sampling] Bin coverage report skipped: {e}")

    # Compute minimal solutions
    results = []
    for N in tqdm(sampled_N, total=len(sampled_N), desc="Processing stratified N"):
        try:
            x, y = chakravala_method(N)
            results.append({"N": N, "x": x, "y": y})
        except ValueError as err:
            print(f"[Warn] N={N}: {err}")

    # Save
    df = pd.DataFrame(results, columns=["N", "x", "y"])
    if df.empty:
        raise SystemExit("No solutions computed; nothing to write.")

    # Cast big ints to text to avoid spreadsheet overflow/precision loss
    df["x"] = df["x"].astype(str)
    df["y"] = df["y"].astype(str)

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_name = f"chakravala_solutions_stratified_{args.low}_to_{args.high}_bins{num_bins}_per{per_bin}_{ts}.xlsx"
    else:
        out_name = args.out

    try:
        df.to_excel(out_name, index=False, engine="xlsxwriter")
    except Exception:
        # Fallback to default engine if xlsxwriter is missing
        df.to_excel(out_name, index=False)
    print(f"Saved {len(df)} solutions to {out_name}")
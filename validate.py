"""
Validate search.py against OEIS A008407: minimal admissible k-tuple
diameters, PROVEN optimal (via exhaustive search) for k <= 342.
https://oeis.org/A008407

Run: python3 validate.py [k_max] [trials_per_k]
"""
import sys
import time
from search import search_many, is_admissible

# OEIS A008407, offset 1: a(1)=0 (k=1) ... a(56)=278 (k=56)
# (truncated here; extend from the OEIS b-file for k up to 342 if needed)
OEIS_A008407 = [
    0, 2, 6, 8, 12, 16, 20, 26, 30, 32, 36, 42, 48, 50, 56, 60, 66, 70, 76,
    80, 84, 90, 94, 100, 110, 114, 120, 126, 130, 136, 140, 146, 152, 156,
    158, 162, 168, 176, 182, 186, 188, 196, 200, 210, 212, 216, 226, 236,
    240, 246, 252, 254, 264, 270, 272, 278,
]


def main():
    k_max = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    k_max = min(k_max, len(OEIS_A008407))
    matches = 0
    t0 = time.time()

    for k in range(2, k_max + 1):
        true_val = OEIS_A008407[k - 1]
        D = true_val + 40
        best_d, best_tup = search_many(k, D, n_trials=trials, seed_start=0)

        if best_d is None:
            print(f"k={k:3d}  true={true_val:4d}  FAILED (no admissible set found, increase D)")
            continue

        ok, bad_p = is_admissible(best_tup)
        status = "MATCH" if best_d == true_val else f"gap=+{best_d - true_val}"
        if not ok:
            status = f"INVALID (fails at p={bad_p}) -- BUG"
        else:
            matches += int(best_d == true_val)

        print(f"k={k:3d}  true={true_val:4d}  found={best_d:4d}  {status}  verified={ok}")

    print(f"\n{matches}/{k_max - 1} exact matches  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()

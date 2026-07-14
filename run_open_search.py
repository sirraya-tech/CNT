"""
Run the search on a specific k in "open" territory (k > 342, where no
diameter is proven optimal -- only best-known upper bounds exist).

Usage:
    python3 run_open_search.py <k> <D_start> <D_end> <D_step> [trials_per_D] [time_budget_sec]

Example (reproduces the k=5511 experiment discussed):
    python3 run_open_search.py 5511 58000 61000 500 25 220
"""
import sys
import time
import json
from sympy import primerange
from search import sieve_search, best_window, is_admissible


def main():
    k = int(sys.argv[1])
    D_start = int(sys.argv[2])
    D_end = int(sys.argv[3])
    D_step = int(sys.argv[4])
    trials_per_D = int(sys.argv[5]) if len(sys.argv) > 5 else 25
    time_budget = float(sys.argv[6]) if len(sys.argv) > 6 else 220.0

    D_values = list(range(D_start, D_end + 1, D_step))
    primes_all = list(primerange(2, D_end + 1))

    best_overall, best_tuple = None, None
    t0 = time.time()
    trial = 0

    for D in D_values:
        primes = [p for p in primes_all if p <= D]
        local_best = None
        for _ in range(trials_per_D):
            surv = sieve_search(D, k, tie_break="random", seed=trial, prime_list=primes)
            trial += 1
            if len(surv) < k:
                continue
            d, tup = best_window(surv, k)
            if d is not None and (local_best is None or d < local_best):
                local_best = d
            if d is not None and (best_overall is None or d < best_overall):
                best_overall, best_tuple = d, tup

        elapsed = time.time() - t0
        print(f"D={D}  local_best={local_best}  elapsed={elapsed:.1f}s  trials={trial}", flush=True)
        if elapsed > time_budget:
            print("time budget reached, stopping")
            break

    print(f"\nBEST DIAMETER FOUND: {best_overall}")
    if best_tuple is not None:
        ok, bad_p = is_admissible(best_tuple)
        print("Independently verified admissible:", ok, "| bad prime (if any):", bad_p)
        if ok:
            out = {"k": k, "diameter": int(best_overall), "tuple": [int(x) for x in best_tuple]}
            fname = f"best_k{k}.json"
            with open(fname, "w") as f:
                json.dump(out, f, indent=2)
            print(f"Saved to {fname}")
        else:
            print("WARNING: result failed independent verification -- discard, do not report.")


if __name__ == "__main__":
    main()

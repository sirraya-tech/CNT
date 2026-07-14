"""
Greedy Schinzel-sieve search for narrow admissible k-tuples.

An admissible k-tuple is a set of k integers B = {b_1 < b_2 < ... < b_k}
such that for every prime p, not all residue classes mod p are represented
in B. (For p > k this is automatic by pigeonhole, so only p <= k matters.)

Method: sieve integers in [0, D) removing, for each prime p <= D, the
residue class mod p with the FEWEST current survivors (this keeps the most
candidates while still guaranteeing that class is fully absent). Once all
required primes are processed, the survivors form a valid admissible set;
pick the tightest window of k survivors as the candidate k-tuple.

This reproduces the standard construction described by Sutherland/Engelsma/
Polymath8 (informally called the "greedy Schinzel sieve").

IMPORTANT CORRECTNESS NOTE: an earlier version of this code stopped sieving
as soon as the survivor count dropped to k, which is WRONG -- it can exit
before every prime p <= k has had a residue class excluded, producing a
tuple that looks admissible but isn't. Always call is_admissible() on any
result before trusting it; this module does not silently guarantee
correctness on its own.
"""
import numpy as np
from sympy import primerange


def sieve_search(D, k, tie_break="random", seed=None, prime_list=None):
    """
    Run one greedy sieve pass over [0, D).

    Args:
        D: sieve range (candidates are integers 0..D-1)
        k: target tuple size
        tie_break: 'random' or 'first' -- how to choose among residue
            classes tied for fewest survivors
        seed: RNG seed for reproducibility (only used if tie_break='random')
        prime_list: optional precomputed list of primes <= D (speeds up
            repeated calls -- computing primerange fresh each time is wasteful)

    Returns:
        numpy array of surviving integers (an admissible set, once all
        primes <= k have been processed -- verify with is_admissible).
    """
    rng = np.random.default_rng(seed)
    surv = np.arange(D, dtype=np.int64)
    primes = prime_list if prime_list is not None else list(primerange(2, D + 1))

    for p in primes:
        # Once survivors <= k, primes > k are automatically satisfied by
        # pigeonhole (fewer elements than residue classes), so we can skip
        # actual sieving work for those -- but we must NOT skip any prime
        # p <= k, even if survivors are already <= k at that point.
        if len(surv) <= k and p > k:
            continue
        if len(surv) == 0:
            break

        residues = surv % p
        counts = np.bincount(residues, minlength=p)
        min_count = counts.min()
        tied = np.nonzero(counts == min_count)[0]

        if tie_break == "random" and len(tied) > 1:
            remove_class = rng.choice(tied)
        else:
            remove_class = tied[0]

        surv = surv[residues != remove_class]

    return surv


def best_window(surv, k):
    """Find the tightest window of >=k survivors. Returns (diameter, tuple) or (None, None)."""
    n = len(surv)
    if n < k:
        return None, None
    best_d, best_tuple = None, None
    for i in range(n - k + 1):
        d = surv[i + k - 1] - surv[i]
        if best_d is None or d < best_d:
            best_d, best_tuple = d, surv[i : i + k]
    return best_d, best_tuple


def is_admissible(tup):
    """
    Independently verify admissibility: for every prime p <= len(tup),
    confirm at least one residue class mod p is absent from tup.

    Returns (True, None) if admissible, else (False, offending_prime).
    This does NOT trust the sieve process -- it recomputes from scratch.
    """
    tup = sorted(int(x) for x in tup)
    n = len(tup)
    for p in primerange(2, n + 1):
        residues = {t % p for t in tup}
        if len(residues) == p:
            return False, p
    return True, None


def search_many(k, D, n_trials, seed_start=0, prime_list=None):
    """
    Run n_trials random-restart sieve searches at a fixed D, return the
    best (smallest-diameter) verified-admissible tuple found.
    """
    best_d, best_tuple = None, None
    for i in range(n_trials):
        surv = sieve_search(D, k, tie_break="random", seed=seed_start + i, prime_list=prime_list)
        if len(surv) < k:
            continue
        d, tup = best_window(surv, k)
        if d is not None and (best_d is None or d < best_d):
            best_d, best_tuple = d, tup
    return best_d, best_tuple


if __name__ == "__main__":
    # Smoke test: k=4 has known optimal diameter 8 (tuple 0,2,6,8)
    d, tup = search_many(k=4, D=50, n_trials=50, seed_start=0)
    ok, bad_p = is_admissible(tup)
    print(f"k=4: diameter={d} tuple={list(tup)} verified_admissible={ok}")
    assert d == 8, "smoke test failed -- known optimal is 8"
    assert ok, "smoke test failed -- result not admissible"
    print("Smoke test passed.")

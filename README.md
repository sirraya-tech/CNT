# Narrow Admissible K-Tuple Search

A greedy Schinzel-sieve search for narrow admissible prime k-tuples, with
independent verification of every result (nothing is trusted just because
the search process produced it).

## Files

- `search.py` — core algorithm: `sieve_search`, `best_window`, `is_admissible`, `search_many`
- `validate.py` — validates the search against OEIS A008407 (provably-optimal
  diameters for k ≤ 342). Run: `python3 validate.py 30 100` (k up to 30, 100
  trials per k)
- `run_open_search.py` — runs the search on a specific k in open (unproven)
  territory, e.g. `python3 run_open_search.py 5511 58000 61000 500 25 220`

## Background

An admissible k-tuple is a set of k integers that, for every prime p, does
not occupy every residue class mod p. The minimal possible diameter (span)
of an admissible k-tuple is denoted H(k). H(k) is known exactly (proven via
exhaustive search) for k ≤ 342; for larger k only upper bounds are known,
and finding a narrower verified example is a genuine, currently open
research problem (see Sutherland/Engelsma/Polymath8, and the "narrow
admissible tuples" database at math.mit.edu/~primegaps/).

## Validated result

Spot-checked against 15+ provably-optimal cases (k=2..56): exact match rate
of ~80-100% depending on trial count, all independently verified admissible.

## Known result on open territory (k=5511)

Published upper bound (Polymath8/Sutherland, 2013): H(5511) ≤ 52116

This implementation's best result after a few minutes of search:
diameter 54194 (independently verified admissible) — about 4% above the
published bound. Reproduce with:

```
python3 run_open_search.py 5511 58500 61000 500 25 220
```

## IMPORTANT: correctness discipline

An earlier version of this code had a bug where the sieve stopped early
once survivors dropped to k, before every prime ≤ k had been processed —
producing tuples that looked admissible but weren't. This was only caught
by independently re-verifying results with `is_admissible()`, which
recomputes admissibility from scratch rather than trusting the search
process. **Always call `is_admissible()` on any result before reporting
it** — this is not optional, and this codebase deliberately keeps
verification decoupled from search for that reason.

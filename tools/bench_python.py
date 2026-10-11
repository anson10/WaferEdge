"""Throughput of the Python module's batch calls (docs/python.md).

    python3 tools/bench_python.py [--maps 20000] [--side 40]

Random round maps (10% fails), one call per function on the whole stack after a warm-up;
best of 3 runs, maps per second, on one thread.
"""

import argparse
import time

import numpy as np

import waferedge as we


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maps", type=int, default=20000)
    parser.add_argument("--side", type=int, default=40)
    parser.add_argument("--rules", default="config/rules.txt")
    args = parser.parse_args()

    n = args.side
    r, c = np.mgrid[0:n, 0:n]
    centre = (n - 1) / 2
    inside = ((r - centre) ** 2 + (c - centre) ** 2 <= (n / 2) ** 2).astype(np.uint8)
    rng = np.random.default_rng(1)
    stack = (inside * (1 + (rng.random((args.maps, n, n)) < 0.1))).astype(np.uint8)
    rules = we.Classifier.load(args.rules)

    print(f"waferedge {we.__version__}, feature backend {we.backend()}, "
          f"{args.maps} maps of {n}x{n}")
    for name, fn in [
        ("features_batch", we.features_batch),
        ("signals", we.signals),
        ("Classifier.classify_batch", rules.classify_batch),
    ]:
        fn(stack[:100])
        best = min(_timed(fn, stack) for _ in range(3))
        print(f"{name:28s} {args.maps / best:12,.0f} maps/s")


def _timed(fn, stack) -> float:
    start = time.perf_counter()
    fn(stack)
    return time.perf_counter() - start


if __name__ == "__main__":
    main()

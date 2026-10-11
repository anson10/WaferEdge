"""WaferEdge's wafer-map signatures, rule classifier and clusters on NumPy arrays.

Maps are 2-D uint8 arrays (0 off-wafer, 1 pass, >= 2 fail), or 3-D stacks of equal-sized
maps for the batch functions. They are read in place (no copy): a wrong dtype or a strided
view raises TypeError; ``np.ascontiguousarray(a, dtype=np.uint8)`` fixes either.
"""

from ._core import (
    PATTERNS,
    SIGNALS,
    Classifier,
    __version__,
    backend,
    clusters,
    features,
    features_batch,
    hough_line,
    read_wmap,
    signals,
)

__all__ = [
    "PATTERNS",
    "SIGNALS",
    "Classifier",
    "__version__",
    "backend",
    "clusters",
    "features",
    "features_batch",
    "hough_line",
    "read_wmap",
    "signals",
]

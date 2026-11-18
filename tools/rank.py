"""Rescores assets/stats.svg with a commit-weighted rank.

Runs after the stats card is generated. Reads the card's own numbers,
recomputes the percentile and rewrites the ring and label in place.
"""

import math
import re
import sys
from pathlib import Path

CARD = Path(sys.argv[1] if len(sys.argv) > 1 else "assets/stats.svg")

# metric: (weight, median, curve)
WEIGHTS = {
    "commits": (0.85, 600, "exp"),
    "prs": (0.075, 20, "exp"),
    "stars": (0.075, 10, "log"),
}
RING = 251.32741228718345  # circumference of the card's r=40 ring


def value(svg, key):
    m = re.search(rf'data-testid="{key}"\s*>([\d,]+)<', svg)
    return int(m.group(1).replace(",", "")) if m else 0


def score(x, median, curve):
    x /= median
    return 1 - 2 ** -x if curve == "exp" else x / (1 + x)


svg = CARD.read_text(encoding="utf-8")
total = sum(w * score(value(svg, k), med, c) for k, (w, med, c) in WEIGHTS.items())
top = round((1 - total) * 100, 1)

svg = re.sub(r'(data-testid="percentile-rank-value"[^>]*>\s*)[\d.]+%', rf"\g<1>{top}%", svg)
svg = re.sub(r"(to\s*\{\s*stroke-dashoffset:\s*)[\d.]+", rf"\g<1>{RING * top / 100}", svg)
CARD.write_text(svg, encoding="utf-8")
print(f"rank: top {top}%")

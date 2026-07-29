"""Mirrored(inner): antithetic pairs. Halves effective N, cancels odd-order terms.

A wrapper, not a fourth strategy.

Not optional either. Mirrored sampling is standard in ES, so it is the honest baseline; a
win measured against unmirrored i.i.d. is a win against a strawman, and much of the easy
variance reduction is already spent by the time coupling gets a turn (docs/00-context.md,
obstacle 1).
"""

# validation

The library checking itself on real hardware. Gate G1 criterion 2 (docs/02): the
simulated-device shortcut every sharding test relies on must not have lied.

- `reference.py` builds one generation per strategy and contraction placement on eight
  simulated CPU devices and writes `reference.json`. `python validation/reference.py`
  regenerates it; `--check` exits non-zero if the committed file differs at all.
- `reference.json` is what `tests/gpu/` holds real accelerators to. The default suite
  rebuilds it and compares within rtol 1e-5
  (`test_the_reference_artifact_still_describes_the_code`), so it cannot go stale
  unnoticed. It did once, by one float32 ulp, for a month.
- `kaggle/t2prime/` runs `tests/gpu` on Kaggle's two T4s. Push it with the Kaggle CLI, or
  with the runner that lives with the experiments:
  `python experiments/phase1/kaggle/run.py validation/kaggle/t2prime`.

Regenerate the reference on CPU after adding a strategy, and only after finding out why
the guard failed if it did.

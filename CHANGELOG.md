# Changelog

## Unreleased

- Runs of a bucket hold at most `2**28` update elements (`MAX_RUN_ELEMENTS`):
  larger buckets step in several runs, bounding the step's transient memory.
- Plain Muon: `Dion(matrix_update=...)` takes a `MatrixUpdate`
  (orthogonalization, `RowNormalization`: NorMuon or none, `MomentumForm`:
  heavy ball or Nesterov) in place of `newton_schulz`; `DION3_UPDATE` is the
  default (the previous update), `MUON_UPDATE` is Keller Jordan's Muon with
  the new `NewtonSchulz.KELLER_TRITON` (`KELLER`: cuBLAS products). Both options are structural: a checkpoint resumes only with the
  options it was saved with, and one saved before them only as NorMuon with
  heavy-ball momentum.
- README: routing fused projections per map (`num_heads`), EMA and weight
  decay (rotation rate), migrating optimizer state to a new route layout, and
  an open question on matrices with low-rank inputs.

## 0.1.0

Initial version.

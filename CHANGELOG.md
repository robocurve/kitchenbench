# Changelog

## Unreleased

### Changed

- **Add time-based `max_seconds` to task specifications.** ([#28](https://github.com/robocurve/kitchenbench/issues/28))
  
  `TaskSpec` now includes a `max_seconds` field representing the real-world completion time budget in seconds (the physical protocol budget, ranging from 60s for simple pick-and-place up to 200s for multi-item sorting). Mock-scale `max_steps` is retained for backwards compatibility.

  Task versions have been bumped (nine tasks bumped to version 3, `scoop_pasta` to version 4) to reflect the metadata change.

- **Scene seeds are now derived from `instance_id` instead of the instance's
  sorted position.** ([#3](https://github.com/robocurve/kitchenbench/issues/3))

  **Results from before this change are not comparable to results after it.**
  Every realized scene is different. Do not compare a policy's score across the
  boundary, and do not aggregate runs from either side of it.

  `tasks.py` seeded each `Scene` with `index`, and `derive_seed` hashes only
  `(eval_seed, scene_seed, epoch)` — task identity never entered the payload. So
  instance *i* of every task drew from a byte-identical PCG64 stream, and 21% of
  the suite's continuous setup draws were exact duplicates across tasks
  (`stack/jitter_x_cm`, `open_container/container_x_cm` and
  `seal_container/lid_offset_cm` were all `+1.0889041282` cm at instance 0).

  Because the tasks shared draws, their scores shared error. Measured over 3000
  eval seeds with a bootstrap CI on the variance ratio, the aggregate score's
  variance was inflated 1.5x to 1.9x in the range where the benchmark actually
  separates policies (score ~0.35–0.50). The 50 trials carried the weight of
  roughly 26–33 independent ones, so confidence intervals computed as if the
  trials were independent were about 40% too narrow.

  Seeding from `instance_id` (unique and task-qualified) takes the distinct scene
  seeds from 5 to 50 and cross-task duplicate draws to zero.

- **All task versions bumped** to mark the scene change as machine-readable, not
  just documented: the nine tasks on `version="1"` are now `"2"`, and
  `scoop_pasta` goes `"2"` -> `"3"`. Every `Scene` stamps `spec.version` into its
  metadata, so without this bump two different scene sets would ship under
  identical version strings — which is the silent non-comparability this change
  exists to prevent.

### Fixed

- `test_canonical_instruction_matches_epoch_zero` recomputed the seed from `index`
  rather than reading `scene.init_seed`, so it pinned a particular seeding scheme
  instead of the invariant it meant to check (displayed instruction == the epoch-0
  realization). It now reads the seed off the `Scene` and is seeding-agnostic.

## Sample Input Layout

This directory now separates runnable toy inputs from imported MAPF benchmark data.

### Directly runnable with current `mapf_pc_lns` CLI

- `empty-16-16.map`
- `agents_goals.txt`
- `kiva-3-500-5.map`
- `0.task`

These match the current loaders in `Instance`:
- generic mode: map + `agents_goals.txt`
- kiva mode: kiva map + `.task`

### Imported MAPF benchmark data

- `benchmark/maps/`
  - `maze-32-32-4.map`
  - `random-32-32-20.map`
- `benchmark/scenarios/scen-even/` (50 files)
- `benchmark/scenarios/scen-random/` (50 files)
- `benchmark/raw_zips/` (original zip archives, kept for traceability)

### Important compatibility note

Current code does **not** load `.scen` files directly.
It currently loads:
- map + `agents_goals.txt` (generic)
- kiva map + `.task` (kiva)

So the benchmark `.scen` files are organized and ready, but they need a conversion step into this project's agent/task format before running.

## Planner Correctness Tests from MAPF Benchmarks

This folder contains lightweight integration tests that:

1. Convert a MAPF benchmark `.scen` file into this project's `agents_goals.txt` format.
2. Run `mapf_pc_lns` on that generated input with:
   - `--lowLevelPlanner mlastar`
   - `--lowLevelPlanner sipps`
3. Assert that each run reports:
   - `Success : true`
   - `Total feasible iterations : > 0`

### Scripts

- `scen_to_agents_goals.sh`
  - Inputs: `<scenario.scen> <agent_count> <output_agents_goals.txt>`
  - Uses the first `agent_count` scenario rows:
    - starts = `(start_x,start_y)`
    - tasks  = `(goal_x,goal_y)`
    - temporal constraints = `0`

- `run_planner_from_scen.sh`
  - Inputs: `<planner_bin> <map_file> <scenario_file> <agent_count> <planner> <artifact_dir>`
  - Runs planner with deterministic seed and fixed LNS parameters.

### Run with CTest

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To see full planner output during tests:

```bash
ctest --test-dir build -V
```

To force full log printing from the test script:

```bash
SHOW_FULL_LOG=1 ctest --test-dir build -V
```

Test artifacts (generated agent files and logs) are written to:

- `build/test_artifacts/`

### Isolated Planner Comparison (Outside LNS Loops)

Use `planner_isolation` to compare low-level planners on a single-goal,
single-agent scenario derived from `.scen`.

```bash
./build/planner_isolation \
  -m ./sample_input/benchmark/maps/maze-32-32-4.map \
  --scen ./sample_input/benchmark/scenarios/scen-even/maze-32-32-4-even-1.scen \
  -k 1 \
  --planner all \
  --seed 42 \
  --printPaths
```

This samples one usable scenario row (controlled by `--seed`) and runs
`mlastar`, `sipps`, and `bfs` on the exact same derived instance.

Notes:

- `-k/--agents` must be `1` (this tool is intentionally single-agent only).
- `--planner` supports: `mlastar`, `sipps`, `bfs`, `both` (`mlastar+sipps`), `all`.
- `--seed 0` uses a time-based seed.

Output includes:

- `soc`
- `makespan`
- `valid_no_conflicts`
- `runtime_sec`
- node expansion/generation stats

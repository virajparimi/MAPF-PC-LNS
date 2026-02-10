#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 <planner_bin> <map_file> <scenario_file> <agent_count> <planner> <artifact_dir>" >&2
  exit 2
fi

planner_bin="$1"
map_file="$2"
scenario_file="$3"
agent_count="$4"
planner_name="$5"
artifact_dir="$6"

if [[ ! -x "$planner_bin" ]]; then
  echo "Planner binary not found/executable: $planner_bin" >&2
  exit 2
fi
if [[ ! -f "$map_file" ]]; then
  echo "Map file not found: $map_file" >&2
  exit 2
fi
if [[ ! -f "$scenario_file" ]]; then
  echo "Scenario file not found: $scenario_file" >&2
  exit 2
fi
if [[ "$planner_name" != "mlastar" && "$planner_name" != "sipps" ]]; then
  echo "planner must be one of: mlastar, sipps (got '$planner_name')" >&2
  exit 2
fi

mkdir -p "$artifact_dir"
agents_file="$artifact_dir/agents_goals_${planner_name}_${agent_count}.txt"
log_file="$artifact_dir/run_${planner_name}_${agent_count}.log"

script_dir="$(cd "$(dirname "$0")" && pwd)"
"$script_dir/scen_to_agents_goals.sh" "$scenario_file" "$agent_count" "$agents_file"

"$planner_bin" \
  -m "$map_file" \
  -a "$agents_file" \
  -k "$agent_count" \
  -l "$agent_count" \
  -t 6 \
  -n 3 \
  -i 40 \
  -d 0 \
  -s greedy \
  -h random \
  -c TA \
  -r absolute \
  --lowLevelPlanner "$planner_name" \
  --seed 42 > "$log_file" 2>&1

if ! grep -qE "Success[[:space:]]*:[[:space:]]*true" "$log_file"; then
  echo "Planner run did not report success=true. Log: $log_file" >&2
  tail -n 60 "$log_file" >&2 || true
  exit 1
fi

if ! grep -qE "Total feasible iterations[[:space:]]*:[[:space:]]*[1-9][0-9]*" "$log_file"; then
  echo "Planner run reported zero feasible iterations. Log: $log_file" >&2
  tail -n 60 "$log_file" >&2 || true
  exit 1
fi

echo "=== Planner Output (${planner_name}) ==="
if [[ "${SHOW_FULL_LOG:-0}" == "1" ]]; then
  cat "$log_file"
else
  awk '
    /^Feasible Solution/ {show=1}
    show {print}
    /^=== Feasible Trajectory ===/ {show=0}
  ' "$log_file"
  awk '
    /^=== Run Summary ===/ {show=1}
    show {print}
    /^=== Low-Level Planner Throughput ===/ {show=0}
  ' "$log_file"
fi

echo "PASS planner=$planner_name agents=$agent_count map=$(basename "$map_file") scen=$(basename "$scenario_file")"

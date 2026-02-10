#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scenario.scen> <agent_count> <output_agents_goal_sequences.txt>" >&2
  exit 2
fi

scen_file="$1"
agent_count="$2"
output_file="$3"

if [[ ! -f "$scen_file" ]]; then
  echo "Scenario file not found: $scen_file" >&2
  exit 2
fi

if ! [[ "$agent_count" =~ ^[0-9]+$ ]] || [[ "$agent_count" -le 0 ]]; then
  echo "agent_count must be a positive integer, got: $agent_count" >&2
  exit 2
fi

available_rows=$(awk 'NR>1{count++} END{print count+0}' "$scen_file")
if [[ "$available_rows" -lt "$agent_count" ]]; then
  echo "Scenario has only $available_rows entries, but $agent_count requested" >&2
  exit 2
fi

# CBS in MAPF-PC can crash on zero-length tasks (start==goal). Filter those.
usable_rows=$(awk 'NR>1 && ($5 != $7 || $6 != $8) {count++} END{print count+0}' "$scen_file")
if [[ "$usable_rows" -lt "$agent_count" ]]; then
  echo "Scenario has only $usable_rows usable entries after filtering start==goal, but $agent_count requested" >&2
  exit 2
fi

{
  echo "$agent_count"
  # MAPF-PC expects tab-separated: num_goals sx sy g1x g1y ...
  awk -v n="$agent_count" 'NR>=2 && ($5 != $7 || $6 != $8) && c<n { printf "1\t%d\t%d\t%d\t%d\n", $5, $6, $7, $8; c++ }' "$scen_file"
  echo "temporal cons:"
} > "$output_file"

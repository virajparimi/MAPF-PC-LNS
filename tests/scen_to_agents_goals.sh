#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <scenario.scen> <agent_count> <output_agents_goals.txt>" >&2
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

{
  echo "$agent_count"
  awk -v n="$agent_count" 'NR>=2 && c<n { print $5 ", " $6; c++ }' "$scen_file"
  echo
  echo "tasks"
  echo "$agent_count"
  awk -v n="$agent_count" 'NR>=2 && c<n { print $7 ", " $8; c++ }' "$scen_file"
  echo
  echo "temporal"
  echo "0"
} > "$output_file"

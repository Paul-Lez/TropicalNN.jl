#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
julia +1.12 --project="$script_dir/.." "$script_dir/distributed_linear_regions.jl"

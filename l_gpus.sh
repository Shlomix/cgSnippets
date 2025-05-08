#!/usr/bin/env bash
# gpu_port_map_simple.sh  –  GPU  PID  PORT  USER
set -euo pipefail

# ── optional:  -m FILE  to override the default map ──
MAP=${1:-}
declare -A M
load() { while read -r u p; do
           [[ $u && $p ]] || continue
           [[ $p =~ ^30([0-9]{2})$ ]] && xy=${BASH_REMATCH[1]} || xy=$(printf "%02d" "$p")
           M[$((3000+10#$xy))]=$u
         done < "$1"; }

if [[ -n $MAP ]]; then load "$MAP"
else
  load <(cat <<EOF
alice 07
bob   87
charlie 02
EOF
)
fi

S=$(mktemp); sudo ss -tunapn >"$S"; trap 'rm -f "$S"' EXIT

echo "GPU PID      PORT USER"
npu-smi info |
awk '/Process name/ {f=1;next} f && $1~/^[0-9]+$/ {print $1,$2}' |
while read -r g p; do
  port=$(awk -v id="pid=$p" '$0~id{for(i=5;i<=6;i++){n=split($i,a,":"); if(a[n]~/^30[0-9][0-9]$/){print a[n];exit}}}' "$S")
  user=${M[$port]:-unknown}
  printf "%-3s %-8s %-5s %s\n" "$g" "$p" "${port:-N/A}" "$user"
done

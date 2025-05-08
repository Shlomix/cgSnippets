#!/usr/bin/env bash
# gpu_port_map_nvsmi.sh  – GPU  PID  PROC‑NAME  PORT  USER
set -euo pipefail

##############################################################################
# 1)  optional map:  ./script -m my.map     (# user  port)
##############################################################################
MAP=${1:-}
declare -A M
load_map () {
  while read -r u p; do
    [[ $u && $p && ! $u =~ ^# ]] || continue
    [[ $p =~ ^30([0-9]{2})$ ]] && xy=${BASH_REMATCH[1]} || xy=$(printf "%02d" "$p")
    M[$((3000+10#$xy))]=$u
  done < "$1"
}
if [[ -n $MAP ]]; then load_map "$MAP"
else
  load_map <(cat <<EOF
alice 07
bob   87
charlie 02
EOF
)
fi

##############################################################################
# 2)  cache sockets once
##############################################################################
S=$(mktemp); sudo ss -tunapn >"$S"; trap 'rm -f "$S"' EXIT

##############################################################################
# 3)  header like nvidia‑smi
##############################################################################
printf '+----+---------+----------------------+-------+--------+\n'
printf '|GPU | PID     | Process name         | PORT  | USER   |\n'
printf '+----+---------+----------------------+-------+--------+\n'

##############################################################################
# 4)  main loop
##############################################################################
npu-smi info |
awk '/Process name/ {f=1; next} f && $2~/^[0-9]+$/ {print $2,$5,$7}' |
while read -r gpu pid pname; do
  port=$(awk -v id="pid=$pid" '$0~id{
        for(i=5;i<=6;i++){n=split($i,a,":");
        if(a[n]~/^30[0-9][0-9]$/){print a[n]; exit}}}' "$S")
  user=${M[$port]:-unknown}
  printf '| %-2s | %-7s | %-20.20s | %-5s | %-6s |\n' \
         "$gpu" "$pid" "$pname" "${port:-N/A}" "$user"
done

printf '+----+---------+----------------------+-------+--------+\n'

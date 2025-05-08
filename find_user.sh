#!/usr/bin/env bash
#
# gpu_port_map.sh  ──  GPU  PID  30xy‑port  USER
#
# Usage:
#   ./gpu_port_map.sh               # uses the embedded default map
#   ./gpu_port_map.sh -m my.map     # use an external map instead
#
# Requires:  npu‑smi  ss  awk  sudo (or CAP_NET_ADMIN)
# ---------------------------------------------------------------------------

set -euo pipefail

##############################################################################
# 1️⃣  Parse CLI
##############################################################################
PORTMAP_FILE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--map) PORTMAP_FILE=$2; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

##############################################################################
# 2️⃣  Load user↔port map into an associative array
##############################################################################
declare -A USERMAP

load_map () {
    local src=$1
    while read -r USER P; do
        [[ -z $USER || $USER =~ ^# ]] && continue        # skip blanks / comments
        if [[ $P =~ ^30([0-9]{2})$ ]]; then
            XY=${BASH_REMATCH[1]}
        else
            XY=$(printf "%02d" "$P")                     # 7  → 07
        fi
        PORT=$((3000 + 10#$XY))                          # 07 → 3007
        USERMAP[$PORT]=$USER
    done < "$src"
}

if [[ -n $PORTMAP_FILE ]]; then
    [[ -r $PORTMAP_FILE ]] || { echo "Cannot read $PORTMAP_FILE"; exit 1; }
    load_map "$PORTMAP_FILE"
else
    # ── Built‑in default map ──  (edit here)
    load_map <(cat <<'EOF'
alice   07
bob     87
charlie 02
EOF
)
fi

##############################################################################
# 3️⃣  Cache socket table once
##############################################################################
sudo ss -tunapn > /tmp/ss.cache.$$

trap 'rm -f /tmp/ss.cache.$$' EXIT

##############################################################################
# 4️⃣  Parse Processes table from npu‑smi and join
##############################################################################
npu-smi info | awk '
    /Process name/ {grab=1; next}
    grab && $1~/^[0-9]+$/ {print $1, $2}
' | while read -r GPU PID; do
    PORT=$(awk -v pid="pid=$PID" '
        $0 ~ pid {
            for(i=5;i<=6;i++){
                n=split($i,a,":");
                if(a[n] ~ /^30[0-9][0-9]$/){print a[n]; exit}
            }
        }' /tmp/ss.cache.$$)

    USER=${USERMAP[$PORT]:-unknown}
    printf "GPU %-3s PID %-7s PORT %-5s USER %s\n" "$GPU" "$PID" "${PORT:-N/A}" "$USER"
done

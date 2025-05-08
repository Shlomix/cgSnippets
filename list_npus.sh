#!/usr/bin/env bash
#
# gpu_port_map.sh  ──  Pretty table view:  GPU | PID | PORT | USER
#   (same CLI as before:  -m /path/to/port_user.map  to override default map)
# ---------------------------------------------------------------------------

set -euo pipefail

##############################################################################
# CLI  (unchanged)  ------------------------------------------------------
##############################################################################
PORTMAP_FILE=""
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--map) PORTMAP_FILE=$2; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

##############################################################################
# Load user↔port map  ----------------------------------------------------
##############################################################################
declare -A USERMAP
load_map () { ... }                 # << same as previous version >>
if [[ -n $PORTMAP_FILE ]]; then
    load_map "$PORTMAP_FILE"
else
    load_map <(cat <<'EOF'
alice   07
bob     87
charlie 02
EOF
)
fi

##############################################################################
# Cache socket table  ----------------------------------------------------
##############################################################################
sudo ss -tunapn > /tmp/ss.cache.$$
trap 'rm -f /tmp/ss.cache.$$' EXIT

##############################################################################
# Collect rows first (GPU PID PORT USER)  -------------------------------
##############################################################################
rows=()
while read -r GPU PID; do
    PORT=$(awk -v pid="pid=$PID" '
        $0 ~ pid {
            for(i=5;i<=6;i++){
                n=split($i,a,":");
                if(a[n] ~ /^30[0-9][0-9]$/){print a[n]; exit}
            }
        }' /tmp/ss.cache.$$)
    USER=${USERMAP[$PORT]:-unknown}
    rows+=("$GPU $PID $PORT $USER")
done < <( npu-smi info | awk '
            /Process name/ {grab=1; next}
            grab && $1~/^[0-9]+$/ {print $1, $2}
         ' )

##############################################################################
# Pretty print  ----------------------------------------------------------
##############################################################################
# Calculate widest USER column
maxu=4                              # at least the width of "USER"
for r in "${rows[@]}"; do
    u=${r##* }                      # last field
    (( ${#u} > maxu )) && maxu=${#u}
done

# Table drawing helpers
line()   { printf '+------+---------+-------+-%*s+\n' "$maxu" '' | tr ' ' '-'; }
header() { printf '| %-4s | %-7s | %-5s | %-*s |\n' "GPU" "PID" "PORT" "$maxu" "USER"; }

line
header
line
for r in "${rows[@]}"; do
    read -r g p pt u <<<"$r"
    printf '| %-4s | %-7s | %-5s | %-*s |\n' "$g" "$p" "${pt:-N/A}" "$maxu" "$u"
done
line

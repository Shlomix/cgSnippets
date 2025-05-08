#!/usr/bin/env bash
#
# gpu_port_map.sh  ──  Print:  GPU  PID  30xy‑port
#
# • Works when the “Processes” table of `npu‑smi info` starts right after the
#   header line that contains the words “Process name”.
# • Needs only the standard tools shipped with every distro (`awk`, `ss`, `grep`).
# • Run with sudo (or give your user CAP_NET_ADMIN so `ss` can see other PIDs).
#
# ---------------------------------------------------------------------------

# 1) Cache the socket table once, because `ss` is the slow part.
sudo ss -tunapn > /tmp/ss.cache

# 2) Parse GPU‑index and PID from the *Processes* table of npu‑smi.
#    The table begins immediately AFTER the line that contains "Process name".
npu-smi info | awk '
    /Process name/ {start=1; next}                # found header → table starts next line
    start && $1 ~ /^[0-9]+$/ {print $1, $2}       # rows whose 1st field is the GPU index
' | while read -r GPU PID; do

    # 3) From the cached ss output pick the *first* 30xy port this PID touches
    PORT=$(awk -v pid="pid=$PID" '
        $0 ~ pid {
            for (i=5; i<=6; i++) {                # check both local (col 5) & peer (col 6)
                n = split($i,a,":")               # split ADDR:PORT → last field is port
                if (a[n] ~ /^30[0-9][0-9]$/) {print a[n]; exit}
            }
        }' /tmp/ss.cache)

    printf "GPU %-3s PID %-7s PORT %s\n" "$GPU" "$PID" "${PORT:-N/A}"
done

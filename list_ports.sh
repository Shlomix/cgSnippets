#!/usr/bin/env bash
# list_npu_ports.sh  –  GPU  PID  30xy-port

sudo ss -tunap > /tmp/ss.cache             # run ss only once

# Pull "GPU‑index  PID" pairs out of the table
npu-smi info | awk '
    /^\| *[0-9]+/ && NF>5 {                # rows that start with  “|  0 …”
        gsub(/^\|/,"",$1)                  # remove leading pipe
        print $1,$5                       # GPU‑index   PID
    }' | while read -r GPU PID; do
        PORT=$(awk -v pid="pid=$PID" '
            $0 ~ pid {
                for (i=5;i<=6;i++){
                    n=split($i,a,":");
                    if (a[n] ~ /^30[0-9][0-9]$/) { print a[n]; exit }
                }
            }' /tmp/ss.cache)
        printf "GPU %s  PID %s  PORT %s\n" "$GPU" "$PID" "$PORT"
done

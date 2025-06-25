wait_for_result() {
    # $1 is unused in your snippet; $2 is the log‑file path
    local pipe=$2
    result="TIMEOUT"                       # default

    # Open the file with tail -F so the FD is re‑opened if the file is
    # truncated or moved by log‑rotation.  Process substitution (< <(…))
    # feeds that stream into read ‑t exactly like your original code.
    while IFS= read -r -t "$RUN_TIMEOUT" line \
          < <(tail -n +0 -F -- "$pipe"); do

        if [[ $line == *"Test Succeeded"* ]]; then
            result="SUCCESS"
            break
        fi
        if [[ $line == *"Test Failed"* ]]; then
            result="FAIL"
            break
        fi
    done
}

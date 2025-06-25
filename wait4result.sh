wait_for_result() {
    local logfile=$2
    local result="TIMEOUT"

    timeout "${RUN_TIMEOUT}s" \
        tail -n +0 -F -- "$logfile" |      # follow across rotations
        while IFS= read -r line; do
            case $line in
                *"Test Succeeded"*) result="SUCCESS"; break ;;
                *"Test Failed"*)    result="FAIL";    break ;;
            esac
        done

    echo "$result"
}

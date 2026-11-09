#!/bin/bash
# measure_run.sh <out file> <command...>
# Runs the command and writes its peak memory in /usr/bin/time -v format ("Maximum resident set size (kbytes): N",
# "Elapsed (wall clock) time ...") to <out file>, which slurm_common.sh record_usage reads. Uses GNU time when it is
# installed, otherwise polls the process's VmHWM (peak RSS, never decreases) every few seconds. Exit code is the
# command's.
out=$1; shift
if [[ -x /usr/bin/time ]]; then
    exec /usr/bin/time -v -o "$out" "$@"
fi
start=$SECONDS
"$@" &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null' TERM INT
peak=0
while kill -0 "$pid" 2>/dev/null; do
    hwm=$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null)
    [[ -n "$hwm" && "$hwm" -gt "$peak" ]] && peak=$hwm
    sleep "${MEASURE_INTERVAL:-5}" & wait $! 2>/dev/null
done
wait "$pid"; rc=$?
{
    echo "Command: $*"
    echo "Maximum resident set size (kbytes): $peak"
    echo "Elapsed (wall clock) seconds: $(( SECONDS - start ))"
    echo "Exit status: $rc"
    echo "(measured by polling VmHWM every ${MEASURE_INTERVAL:-5}s, GNU time not installed)"
} > "$out"
exit $rc

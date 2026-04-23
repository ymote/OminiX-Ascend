#!/usr/bin/env bash
# QIE-Q2.2 Q4_1-resident Gate 0 probe — build + run on ac03.
# Reproduction:  bash build_and_run_q4_1.sh
# Exit codes:
#   0 = GREEN  (op accepts W4 + per-group antiquantOffset, cos_sim > 0.99)
#   1 = YELLOW (op works but numerics or perf off — escalate to PM)
#   2 = RED    (op rejects W4+offset — fall back to F16 for Q4_1 tensors)
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

# Cohabit with Agent A4c — contract mandates the ac03 HBM lock.
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
    echo "[probe] HBM lock present at $LOCK — waiting..."
    while [ -e "$LOCK" ]; do sleep 5; done
fi
echo "qie_q2_q4_1_probe $$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

g++ -std=c++17 -O2 -o test_qie_q4_1_probe test_qie_q4_1_probe.cpp \
    -I$ASCEND_TOOLKIT_HOME/aarch64-linux/include \
    -L$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64 \
    -lascendcl -lopapi -lnnopbase -ldl

echo "--- build OK ---"
./test_qie_q4_1_probe
rc=$?
echo "--- probe exit rc=$rc ---"
exit $rc

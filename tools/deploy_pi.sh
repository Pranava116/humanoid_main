#!/usr/bin/env bash
# Copy the trained walking policy to the robot's Pi over SSH and check that it loads there.
#
#   tools/deploy_pi.sh pi@10.144.86.201            # default repo path ~/humanoid_main
#   tools/deploy_pi.sh pi@10.144.86.201 --pull     # also git pull the code on the Pi first
#   PI_REPO=~/other/path tools/deploy_pi.sh user@host
#
# The .onnx file is not in git (*.onnx is ignored), so it always goes over scp.
# The code (robot/rl, config/stack/rl.yaml) comes with git: push on the Mac, pull on the Pi.
set -euo pipefail

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    echo
    echo "options: --pull     git pull on the Pi before copying"
    echo "         --policy F local policy file (default models/policies/walk.onnx)"
    echo "env:     PI_REPO    repo path on the Pi (default ~/humanoid_main)"
}

HOST=""
PULL=0
POLICY="models/policies/walk.onnx"
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --pull) PULL=1 ;;
        --policy) POLICY="$2"; shift ;;
        -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
        *) HOST="$1" ;;
    esac
    shift
done
[ -n "$HOST" ] || { echo "missing user@host" >&2; usage >&2; exit 2; }

cd "$(dirname "$0")/.."
[ -f "$POLICY" ] || { echo "policy not found: $POLICY" >&2; exit 1; }
REPO="${PI_REPO:-~/humanoid_main}"

echo "== 1/4 connect to $HOST"
ssh -o ConnectTimeout=10 "$HOST" "test -d $REPO" || { echo "repo $REPO not found on the Pi" >&2; exit 1; }

if [ "$PULL" = 1 ]; then
    echo "== 2/4 git pull on the Pi"
    ssh "$HOST" "cd $REPO && git pull --ff-only"
else
    echo "== 2/4 (skipping git pull; pass --pull to update the code)"
fi

echo "== 3/4 copy $POLICY ($(du -h "$POLICY" | cut -f1))"
ssh "$HOST" "mkdir -p $REPO/models/policies"
scp "$POLICY" "$HOST:$REPO/models/policies/walk.onnx"

echo "== 4/4 check onnxruntime + policy on the Pi"
ssh "$HOST" "cd $REPO && PY=python3; [ -x .venv/bin/python ] && PY=.venv/bin/python;
  \$PY -c 'import onnxruntime' 2>/dev/null || { echo 'installing onnxruntime...'; \$PY -m pip install 'onnxruntime>=1.16'; };
  \$PY -c '
from robot.config import load_config
from robot.rl.policy import OnnxPolicy
import numpy as np
cfg = load_config()
p = OnnxPolicy(cfg.path(str(cfg.get(\"rl.policy_file\", \"models/policies/walk.onnx\"))))
a = p.act(np.zeros(p.obs_dim))
print(\"policy OK:\", p.obs_dim, \"inputs ->\", len(a), \"joints\")
print(\"walking controller:\", cfg.get(\"behavior.walking.controller\"))
'"
echo "done. On the Pi: python -m robot.run --backend real"

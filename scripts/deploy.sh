#!/usr/bin/env bash
# Deploy Earwitness to a fresh Ubuntu server over SSH (password auth).
#
#   bash scripts/deploy.sh [user@host]                      # with the deploy key
#   SSHPASS='server-password' bash scripts/deploy.sh         # first time, no key yet
#
# A password is only read from $SSHPASS (for SSH and sudo) and never written
# to disk. With the key, sudo must be passwordless on the server. Re-running the script updates the code
# and restarts the service.
set -euo pipefail

TARGET="${1:-linuxuser@78.141.206.107}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${DEPLOY_KEY:-$HOME/.ssh/earwitness_deploy}"

# Prefer the deploy key (no password needed); fall back to SSHPASS.
if [[ -f "$KEY" ]] && ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$KEY" "$TARGET" true 2>/dev/null; then
  SSH_OPTS=(-o BatchMode=yes -i "$KEY")
  RUN=()
  SUDO_LINE=""
elif [[ -n "${SSHPASS:-}" ]]; then
  command -v sshpass >/dev/null || brew install hudochenkov/sshpass/sshpass
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no -o PreferredAuthentications=password)
  RUN=(sshpass -e)
  SUDO_LINE="$SSHPASS"
else
  echo "No working deploy key ($KEY) and no SSHPASS set." >&2
  exit 1
fi

echo "==> Copying code to $TARGET:~/earwitness"
"${RUN[@]}" rsync -az --delete \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude .venv --exclude web/node_modules --exclude web/dist --exclude web/dist-stable \
  --exclude cases --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache \
  --exclude .git --exclude .DS_Store \
  "$ROOT/" "$TARGET:earwitness/"

echo "==> Setting up the server (this takes a few minutes the first time)"
# bash -s reads its script from stdin line by line: the first line reads the
# next line (the sudo password) into a variable, then the setup script follows.
{ printf 'read -r SUDO_PASS\n%s\n' "$SUDO_LINE"; cat "$ROOT/scripts/remote_setup.sh"; } |
  "${RUN[@]}" ssh "${SSH_OPTS[@]}" "$TARGET" 'bash -s'

HOST="${TARGET#*@}"
echo "==> Done: http://$HOST/"

#!/usr/bin/env bash
# Copy files between this machine and the pod over SSH (rsync; reruns only send what changed).
#   up:   the repo (no .venv, runs, dist, data), data/pages, and the macOS pages data/pages_ax and data/pages_s2ax when present. Pages are already .json.gz, so nothing is recompressed.
#   down: teacher outputs, the teacher cache, assembled examples, runs/ and dist/.
# usage: gpu/sync.sh up|down HOST PORT   (RunPod shows: ssh root@HOST -p PORT -i ~/.ssh/id_ed25519)
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -eq 3 ] || { echo "usage: $0 up|down HOST PORT" >&2; exit 2; }
direction=$1 host=$2 port=$3
remote_dir=${REMOTE_DIR:-/workspace/tiny-grounder}
ssh_cmd="ssh -p $port -i ${SSH_KEY:-$HOME/.ssh/id_ed25519} -o StrictHostKeyChecking=accept-new"
remote=root@$host:$remote_dir
down_dirs=(data/teacher data/teacher_cache data/examples runs dist)

# shellcheck disable=SC2086
$ssh_cmd "root@$host" "mkdir -p $remote_dir/data/pages ${down_dirs[*]/#/$remote_dir/}"
case $direction in
  up)
    rsync -a --partial -e "$ssh_cmd" --exclude /.git --exclude /.venv --exclude /runs --exclude /dist \
      --exclude /data --exclude __pycache__ --exclude .pytest_cache ./ "$remote/"
    rsync -a --partial -e "$ssh_cmd" data/pages/ "$remote/data/pages/"
    for native in pages_ax pages_s2ax; do
      [ -d "data/$native" ] || continue
      $ssh_cmd "root@$host" "mkdir -p $remote_dir/data/$native"
      rsync -a --partial -e "$ssh_cmd" --exclude .DS_Store "data/$native/" "$remote/data/$native/"
    done
    ;;
  down)
    for dir in "${down_dirs[@]}"; do
      mkdir -p "$dir"
      rsync -a --partial -e "$ssh_cmd" "$remote/$dir/" "$dir/"
    done
    ;;
  *)
    echo "direction must be up or down, got '$direction'" >&2
    exit 2
    ;;
esac

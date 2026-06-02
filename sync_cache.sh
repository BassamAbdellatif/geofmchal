#!/bin/bash
# sync_cache.sh — replicate the 7A float16 tile cache to each compute node's
# local disk, so every node reads from its own disk (→ own OS page cache) during
# the Phase-5 campaign instead of hammering the head node's NFS in parallel.
#
# The cache is node-agnostic (deterministic preprocessed fold-0 arrays), so it is
# built once on the head and copied verbatim.
#
# Usage:
#   ./sync_cache.sh "host1 host2 host3"            # uses default src/dst
#   ./sync_cache.sh "n1 n2 n3" /scratch/cache7a    # custom dst on each node
#   SRC=/path/to/cache7a ./sync_cache.sh "n1 n2"   # custom source
#
# Then on each node, train with:  --cache-dir <DST>
# Cleanup later:  for h in $HOSTS; do ssh $h 'rm -rf <DST>'; done

set -euo pipefail

SRC="${SRC:-/mnt/head/users/bassam/data/geofmdata/cache7a}"
HOSTS="${1:-}"
DST="${2:-/scratch/cache7a}"

if [[ -z "$HOSTS" ]]; then
    echo "Usage: $0 \"host1 host2 ...\" [dest_dir_on_node]" >&2
    echo "  (SRC env var overrides the source; default: $SRC)" >&2
    exit 1
fi

if [[ ! -f "$SRC/fold0_train.done" || ! -f "$SRC/fold0_val.done" ]]; then
    echo "WARNING: $SRC is missing a .done flag — the cache may be incomplete." >&2
    echo "         Build/verify it first, then re-run this script." >&2
    exit 1
fi

SRC_SIZE=$(du -sh "$SRC" | cut -f1)
echo "Source: $SRC ($SRC_SIZE)"
echo "Dest:   <node>:$DST"
echo "Hosts:  $HOSTS"
echo

for h in $HOSTS; do
    echo "=== $h ==="
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" true 2>/dev/null; then
        echo "  [skip] cannot SSH to $h (passwordless SSH required)"
        continue
    fi
    ssh "$h" "mkdir -p '$DST'"
    # -a archive, --partial resume, only copy changed files (idempotent re-runs).
    rsync -a --partial --info=progress2 "$SRC/" "$h:$DST/"
    # Verify both .done flags landed.
    if ssh "$h" "test -f '$DST/fold0_train.done' && test -f '$DST/fold0_val.done'"; then
        echo "  [ok] $h:$DST ready"
    else
        echo "  [FAIL] $h:$DST missing .done flags after sync"
    fi
done

echo
echo "Done. On each node, pass:  --cache-dir $DST"

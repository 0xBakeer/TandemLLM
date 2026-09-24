#!/bin/bash
# Make the access tokens (SRV-31) for the :8000 service, once:
#
#   bash ops/make-secrets.sh            creates ~/.qwen38-spark-engine/secrets.env (mode 600)
#   bash ops/make-secrets.sh --rotate   replaces both tokens (every dashboard session ends)
#
# QSE_ADMIN_TOKEN opens the dashboard and the cache routes; QSE_METRICS_TOKEN is for Prometheus
# (the k8s Secret `qse-metrics-token`, OPS-20). 32 random bytes each, hex. The values are never
# printed: read them with `cat` on the box when a client needs one. ops/start.sh sources the file.
set -eu
DIR="${QSE_STATE_DIR:-$HOME/.qwen38-spark-engine}"
FILE="$DIR/secrets.env"
if [ -f "$FILE" ] && [ "${1:-}" != "--rotate" ]; then
    echo "[secrets] $FILE exists; --rotate replaces it"; exit 0
fi
umask 077
mkdir -p "$DIR"
tok() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
TMP="$FILE.tmp.$$"
{
    echo "# access tokens for the qwen38-spark-engine :8000 service (SRV-31); made $(date -Iseconds)"
    echo "QSE_ADMIN_TOKEN=$(tok)"
    echo "QSE_METRICS_TOKEN=$(tok)"
} > "$TMP"
chmod 600 "$TMP"
mv "$TMP" "$FILE"
echo "[secrets] wrote $FILE (mode 600); restart the service to use it"

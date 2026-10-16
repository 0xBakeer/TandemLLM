#!/usr/bin/env bash
# The README's and the gallery's pictures: PNGs from e2e/readme-shots.spec.ts to WebP under docs/gallery/.
#   scripts/readme-shots.sh [png-dir]     (default: screenshots/readme)
# Needs cwebp (libwebp). Fails when a picture is 400 KB or more.
set -euo pipefail
cd "$(dirname "$0")/.."
src=${1:-screenshots/readme}
out=../docs/gallery
max=$((400 * 1024))
command -v cwebp >/dev/null || { echo "cwebp not found (libwebp)" >&2; exit 1; }
mkdir -p "$out"
status=0
for png in "$src"/*.png; do
  name=$(basename "$png" .png)
  cwebp -quiet -q 82 -m 6 -metadata none "$png" -o "$out/$name.webp"
  size=$(wc -c < "$out/$name.webp" | tr -d ' ')
  printf '%-24s %7d bytes\n' "$name.webp" "$size"
  if [ "$size" -ge "$max" ]; then echo "  $name.webp is over 400 KB" >&2; status=1; fi
done
exit $status

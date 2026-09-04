#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/data/san_francisco"}
ARCHIVE_URL=${ARCHIVE_URL:-"https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_AIRSAR.zip"}
LABEL_URL=${LABEL_URL:-"https://raw.githubusercontent.com/liuxuvip/PolSF/master/SF-AIRSAR/SF-AIRSAR-label2d.png"}

mkdir -p "$DATA_DIR"
archive=$(mktemp --suffix=.zip)
trap 'rm -f "$archive"' EXIT

echo "Downloading AIRSAR archive..."
curl --fail --location --retry 3 "$ARCHIVE_URL" --output "$archive"
unzip -p "$archive" \
  SAN_FRANCISCO_AIRSAR/san_francisco900x1024.stk \
  > "$DATA_DIR/san_francisco900x1024.stk"

echo "Downloading label map..."
curl --fail --location --retry 3 "$LABEL_URL" \
  --output "$DATA_DIR/SF-AIRSAR-label2d.png"

expected_stk_size=$((900 * 1024 * 10))
actual_stk_size=$(stat --format=%s "$DATA_DIR/san_francisco900x1024.stk")
if [[ "$actual_stk_size" -ne "$expected_stk_size" ]]; then
  echo "Unexpected STK size: $actual_stk_size (expected $expected_stk_size)" >&2
  exit 1
fi

echo "San Francisco AIRSAR data is ready in $DATA_DIR"

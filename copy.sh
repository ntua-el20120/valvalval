#!/usr/bin/env bash
# copy_dirs.sh — Recursively copy all subdirs of a given path, skipping any dir named 'arrays'

set -euo pipefail

# --- Usage and input checks ---
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/source" >&2
  exit 1
fi

src=$(realpath "$1")
[[ -d $src ]] || { echo "Not a valid directory: $src" >&2; exit 1; }

dest=$PWD

# --- Find all directories that are NOT named 'arrays' or inside one ---
find "$src" -type d -name arrays -prune -o -type d -print | while IFS= read -r dir; do
  rel=${dir#"$src"/}
  [[ "$rel" == "$src" ]] && continue  # skip the root itself
  mkdir -p "$dest/$rel"
done

# --- Copy files that are NOT inside any 'arrays' directory ---
find "$src" -type d -name arrays -prune -o -type f -print | while IFS= read -r file; do
  rel=${file#"$src"/}
  mkdir -p "$dest/$(dirname "$rel")"
  cp -a "$file" "$dest/$rel"
done

echo "Done. All directories and files copied, skipping any named 'arrays'."

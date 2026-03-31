#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT_ROOT="${ROOT_DIR}/data/af_subset"
DEFAULT_TARGETS_CSV="${ROOT_DIR}/data/Proteinas_secuencias.csv"
AWS_BIN="${AWS_BIN:-aws}"

JSON_S3_URI="s3://openfold/benchmarking_data/input_jsons_converted/wo_templates/fb_protein.json"
MSA_S3_PREFIX="s3://openfold/benchmarking_data/msas/foldbench_msas"
CIF_S3_PREFIX="s3://openfold/benchmarking_data/reference_structures/foldbench_protein"

OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
TARGETS_FILE=""
TARGETS_CSV=""
LIMIT=""
LIST_ONLY=0
SKIP_EXISTING=1
DOWNLOAD_JSON=1
DOWNLOAD_MSAS=1
DOWNLOAD_STRUCTURES=1

usage() {
  cat <<'EOF'
Usage: data/download_data.sh [options]

Download a Foldbench/OpenFold subset using either:
  - a TXT file with target IDs like 7qrj_A, or
  - the checked-in CSV manifest generated from Colab.

Options:
  --output-root PATH      Download root. Default: data/af_subset
  --targets-file PATH     Plain-text file with one target per line.
  --targets-csv PATH      CSV manifest with an `msa_dir_name` column.
  --limit N               Keep only the first N unique targets.
  --list-targets          Print resolved targets and exit.
  --no-skip-existing      Re-download files even if they already exist.
  --skip-json             Skip fb_protein.json.
  --skip-msas             Skip MSA download.
  --skip-structures       Skip reference structure download.
  -h, --help              Show this help message.
EOF
}

log() {
  printf '[download] %s\n' "$*"
}

die() {
  printf '[download] ERROR: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --targets-file)
      TARGETS_FILE="$2"
      shift 2
      ;;
    --targets-csv)
      TARGETS_CSV="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --list-targets)
      LIST_ONLY=1
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --skip-json)
      DOWNLOAD_JSON=0
      shift
      ;;
    --skip-msas)
      DOWNLOAD_MSAS=0
      shift
      ;;
    --skip-structures)
      DOWNLOAD_STRUCTURES=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

require_cmd awk
require_cmd cut
require_cmd grep
require_cmd head
require_cmd mktemp
require_cmd sed
require_cmd sort
require_cmd tail
require_cmd tr

if [[ -z "$TARGETS_FILE" && -z "$TARGETS_CSV" && -f "$DEFAULT_TARGETS_CSV" ]]; then
  TARGETS_CSV="$DEFAULT_TARGETS_CSV"
fi

if [[ -n "$TARGETS_FILE" && -n "$TARGETS_CSV" ]]; then
  die "Use either --targets-file or --targets-csv, not both."
fi

if [[ -z "$TARGETS_FILE" && -z "$TARGETS_CSV" ]]; then
  die "No targets source provided. Pass --targets-file or --targets-csv."
fi

load_targets() {
  if [[ -n "$TARGETS_FILE" ]]; then
    [[ -f "$TARGETS_FILE" ]] || die "Targets file not found: $TARGETS_FILE"
    sed '/^[[:space:]]*$/d' "$TARGETS_FILE"
    return
  fi

  [[ -f "$TARGETS_CSV" ]] || die "Targets CSV not found: $TARGETS_CSV"
  tail -n +2 "$TARGETS_CSV" | cut -d, -f3 | sed '/^[[:space:]]*$/d'
}

mapfile -t TARGETS < <(load_targets | sort -u | { if [[ -n "$LIMIT" ]]; then head -n "$LIMIT"; else cat; fi; })
(( ${#TARGETS[@]} > 0 )) || die "Resolved target list is empty."

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
JSON_DIR="${OUTPUT_ROOT}/jsons"
MSA_DIR="${OUTPUT_ROOT}/foldbench_msas"
CIF_DIR="${OUTPUT_ROOT}/reference_structures"
META_DIR="${OUTPUT_ROOT}/meta"

mkdir -p "$JSON_DIR" "$MSA_DIR" "$CIF_DIR" "$META_DIR"
printf '%s\n' "${TARGETS[@]}" > "${META_DIR}/selected_targets.txt"

if (( LIST_ONLY )); then
  printf '%s\n' "${TARGETS[@]}"
  exit 0
fi

require_cmd "$AWS_BIN"

if (( DOWNLOAD_JSON )); then
  log "Downloading Foldbench JSON manifest"
  "$AWS_BIN" s3 cp "$JSON_S3_URI" "$JSON_DIR/" --no-sign-request
fi

if (( DOWNLOAD_MSAS )); then
  for target in "${TARGETS[@]}"; do
    destination="${MSA_DIR}/${target}"
    if (( SKIP_EXISTING )) && [[ -f "${destination}/cfdb_hits.a3m" ]]; then
      log "Skipping MSA for ${target}; cfdb_hits.a3m already exists"
      continue
    fi

    log "Downloading MSA for ${target}"
    "$AWS_BIN" s3 cp "${MSA_S3_PREFIX}/${target}/" "${destination}/" --recursive --no-sign-request
  done
fi

if (( DOWNLOAD_STRUCTURES )); then
  CIF_INDEX="$(mktemp)"
  trap 'rm -f "$CIF_INDEX"' EXIT

  log "Indexing available reference structures"
  "$AWS_BIN" s3 ls "$CIF_S3_PREFIX/" --no-sign-request | awk '{print $4}' > "$CIF_INDEX"

  missing_structures=0
  for target in "${TARGETS[@]}"; do
    pdb_id="${target%%_*}"
    pdb_id="$(printf '%s' "$pdb_id" | tr '[:upper:]' '[:lower:]')"
    cif_key="$(grep -E "^${pdb_id}-assembly1_.*\.cif$" "$CIF_INDEX" | head -n 1 || true)"

    if [[ -z "$cif_key" ]]; then
      log "No reference structure found for ${target}"
      missing_structures=$((missing_structures + 1))
      continue
    fi

    if (( SKIP_EXISTING )) && [[ -f "${CIF_DIR}/${cif_key}" ]]; then
      log "Skipping structure ${cif_key}; file already exists"
      continue
    fi

    log "Downloading structure ${cif_key}"
    "$AWS_BIN" s3 cp "${CIF_S3_PREFIX}/${cif_key}" "${CIF_DIR}/${cif_key}" --no-sign-request
  done

  log "Structure misses: ${missing_structures}"
fi

log "Done."
log "JSON dir: ${JSON_DIR}"
log "MSA dir: ${MSA_DIR}"
log "CIF dir: ${CIF_DIR}"

#!/bin/bash
set -euo pipefail

ORIG_JSON=""
JSON_OVERRIDES=()
YAML_OVERRIDES=()
DEBUG=false
SAVE_DIR=""
LOAD_DIR=""
DRY_RUN=false
SCRIPT_ARGS=()

# Check if Go-based yq is available
HAS_YQ=false
if command -v yq >/dev/null && yq --version 2>&1 | grep -qi 'mikefarah'; then
  HAS_YQ=true
fi

# Fallback for YAML editing using Python
edit_yaml_py() {
  local file="$1"
  local key_path="$2"
  local new_value="$3"
  python3 - <<EOF
import yaml
with open("$file") as f:
    data = yaml.safe_load(f)

def set_nested(d, keys, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

set_nested(data, "$key_path".split("."), yaml.safe_load("""$new_value"""))

with open("$file", "w") as f:
    yaml.safe_dump(data, f)
EOF
}

# Parse wrapper arguments until `--`
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      ORIG_JSON="$2"
      shift 2
      ;;
    --json-override)
      JSON_OVERRIDES+=("$2")
      shift 2
      ;;
    --yaml-override)
      YAML_OVERRIDES+=("$2")
      shift 2
      ;;
    --save-dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --load-dir)
      LOAD_DIR="$2"
      shift 2
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --)
      shift
      SCRIPT_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -n "$LOAD_DIR" ]]; then
  TEMP_JSON="$LOAD_DIR/config.json"
  TEMP_YAML="$LOAD_DIR/config.yaml"
  if [[ ! -f "$TEMP_JSON" || ! -f "$TEMP_YAML" ]]; then
    echo "Error: Directory '$LOAD_DIR' must contain config.json and config.yaml"
    exit 1
  fi
  echo "Using existing configs from: $LOAD_DIR"
else
  if [[ -z "$ORIG_JSON" ]]; then
    echo "Must provide -m path/to/config.json"
    exit 1
  fi

  TEMP_JSON=$(mktemp --suffix=.json)
  TEMP_YAML=$(mktemp --suffix=.yaml)

  cp "$ORIG_JSON" "$TEMP_JSON"

  YAML_PATH=$(jq -r '.yaml_path' "$TEMP_JSON")
  cp "$YAML_PATH" "$TEMP_YAML"

  for expr in "${YAML_OVERRIDES[@]}"; do
    key="${expr%%=*}"
    val="${expr#*=}"
    if [[ "$HAS_YQ" == true ]]; then
      yq eval ".$key = $val" -i "$TEMP_YAML"
    else
      edit_yaml_py "$TEMP_YAML" "$key" "$val"
    fi
  done

  jq --arg new_yaml "$TEMP_YAML" '.yaml_path = $new_yaml' "$TEMP_JSON" > "${TEMP_JSON}.mod"
  mv "${TEMP_JSON}.mod" "$TEMP_JSON"

  for expr in "${JSON_OVERRIDES[@]}"; do
    jq "$expr" "$TEMP_JSON" > "${TEMP_JSON}.mod"
    mv "${TEMP_JSON}.mod" "$TEMP_JSON"
  done

  if [[ -n "$SAVE_DIR" ]]; then
    TS=$(date +"%Y%m%d_%H%M%S")
    OUT_DIR="${SAVE_DIR}_${TS}"
    mkdir -p "$OUT_DIR"
    cp "$TEMP_JSON" "$OUT_DIR/config.json"
    cp "$TEMP_YAML" "$OUT_DIR/config.yaml"
    echo "Saved config files to: $OUT_DIR"
    TEMP_JSON="$OUT_DIR/config.json"
    TEMP_YAML="$OUT_DIR/config.yaml"
  fi
fi

if [[ "$DEBUG" == true ]]; then
  echo ""
  echo "JSON Config ($TEMP_JSON):"
  jq . "$TEMP_JSON"
  echo ""
  echo "YAML Config ($TEMP_YAML):"
  cat "$TEMP_YAML"
  echo ""
fi

COMMAND=(./sys_run.sh -m "$TEMP_JSON" "${SCRIPT_ARGS[@]}")

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run mode - command that would be run:"
  printf '%q ' "${COMMAND[@]}"
  echo
else
  "${COMMAND[@]}"
fi

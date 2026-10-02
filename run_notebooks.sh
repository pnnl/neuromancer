#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_notebooks.sh [kernel] [repo_dir] [timeout_seconds] [notebook_list_file]
#
# notebook_list_file, when provided, must be a NUL-delimited list of notebook paths.
# This is used by PR CI to execute only notebooks added by the pull request.

KERNEL="${1:-python3}"
REPO_DIR="${2:-.}"
EXEC_TIMEOUT_SECONDS="${3:-3600}"
NOTEBOOK_LIST_FILE="${4:-}"
NOTEBOOK_LOG_TAIL_LINES="${NOTEBOOK_LOG_TAIL_LINES:-80}"

if [[ "$EXEC_TIMEOUT_SECONDS" != "-1" && ! "$EXEC_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXEC_TIMEOUT_SECONDS must be -1 or a positive integer." >&2
  exit 2
fi

if [[ ! "$NOTEBOOK_LOG_TAIL_LINES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NOTEBOOK_LOG_TAIL_LINES must be a positive integer." >&2
  exit 2
fi

if [[ "$EXEC_TIMEOUT_SECONDS" != "-1" ]] && ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: timeout command is required when EXEC_TIMEOUT_SECONDS is positive." >&2
  exit 2
fi

if ! command -v jupyter >/dev/null 2>&1; then
  echo "ERROR: jupyter is not available on PATH." >&2
  exit 2
fi

if ! REPO_DIR_ABS="$(cd "$REPO_DIR" && pwd -P)"; then
  echo "ERROR: repository directory does not exist: $REPO_DIR" >&2
  exit 2
fi

cd "$REPO_DIR_ABS"

TMP_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
TMP_DIR="$(mktemp -d "$TMP_ROOT/notebook-test.XXXXXX")"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

if [[ -n "${NOTEBOOK_LOG_DIR:-}" ]]; then
  mkdir -p "$NOTEBOOK_LOG_DIR"
fi

selected_notebooks=()
excluded_notebooks=()
ci_skip_notebooks=()
ci_skipped_notebooks=()
missing_notebooks=()
passed_notebooks=()
failed_notebooks=()
timed_out_notebooks=()

normalize_path() {
  local path="$1"

  if [[ "$path" == "$REPO_DIR_ABS"/* ]]; then
    path="${path#"$REPO_DIR_ABS"/}"
  elif [[ "$path" == /* ]]; then
    echo "ERROR: notebook path is outside the repository: $path" >&2
    exit 2
  fi

  path="${path#./}"
  printf '%s\n' "$path"
}

is_excluded_notebook() {
  local path="$1"
  local dir_part=""
  local part
  local generated_dir
  local generated_dirs=(
    ".ipynb_checkpoints"
    "_build"
    "build"
    "dist"
    "__pycache__"
    "htmlcov"
    "lightning_logs"
    "site"
    "target"
  )

  [[ "$path" == *.ipynb ]] || return 0

  if [[ "$path" == */* ]]; then
    dir_part="${path%/*}"
  fi

  if [[ -n "$dir_part" ]]; then
    IFS='/' read -r -a path_parts <<< "$dir_part"
    for part in "${path_parts[@]}"; do
      [[ -n "$part" ]] || continue

      if [[ "$part" == .* ]]; then
        return 0
      fi

      for generated_dir in "${generated_dirs[@]}"; do
        if [[ "$part" == "$generated_dir" ]]; then
          return 0
        fi
      done
    done
  fi

  return 1
}

load_ci_skip_list() {
  local line
  local path

  [[ -n "${NOTEBOOK_CI_SKIP_NOTEBOOKS:-}" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -n "$line" ]] || continue
    path="$(normalize_path "$line")"
    ci_skip_notebooks+=("$path")
  done <<< "$NOTEBOOK_CI_SKIP_NOTEBOOKS"
}

is_ci_skipped_notebook() {
  local path="$1"
  local skipped

  for skipped in "${ci_skip_notebooks[@]}"; do
    if [[ "$path" == $skipped ]]; then
      return 0
    fi
  done

  return 1
}

add_candidate() {
  local raw_path="$1"
  local path

  path="$(normalize_path "$raw_path")"
  [[ -n "$path" ]] || return 0

  if is_excluded_notebook "$path"; then
    excluded_notebooks+=("$path")
    return 0
  fi

  if is_ci_skipped_notebook "$path"; then
    ci_skipped_notebooks+=("$path")
    return 0
  fi

  if [[ ! -f "$path" ]]; then
    missing_notebooks+=("$path")
    return 0
  fi

  selected_notebooks+=("$path")
}

discover_full_suite() {
  local discovered=()
  local discovered_file="$TMP_DIR/discovered.zlist"
  local sorted_file="$TMP_DIR/discovered.sorted.zlist"
  local notebook

  find . -mindepth 1 \
    \( -type d \( \
      -name ".*" -o \
      -name "_build" -o \
      -name "build" -o \
      -name "dist" -o \
      -name "__pycache__" -o \
      -name "htmlcov" -o \
      -name "lightning_logs" -o \
      -name "site" -o \
      -name "target" \
    \) -prune \) -o \
    \( -type f -name "*.ipynb" -print0 \) \
    > "$discovered_file"

  LC_ALL=C sort -z "$discovered_file" > "$sorted_file"
  mapfile -d '' -t discovered < "$sorted_file"

  for notebook in "${discovered[@]}"; do
    add_candidate "$notebook"
  done
}

read_explicit_list() {
  local listed=()
  local notebook

  if [[ ! -f "$NOTEBOOK_LIST_FILE" ]]; then
    echo "ERROR: notebook list file does not exist: $NOTEBOOK_LIST_FILE" >&2
    exit 2
  fi

  mapfile -d '' -t listed < "$NOTEBOOK_LIST_FILE"

  for notebook in "${listed[@]}"; do
    add_candidate "$notebook"
  done
}

sort_selected_notebooks() {
  local selected_file="$TMP_DIR/selected.zlist"
  local sorted_file="$TMP_DIR/selected.sorted.zlist"

  if [[ "${#selected_notebooks[@]}" -gt 0 ]]; then
    printf '%s\0' "${selected_notebooks[@]}" > "$selected_file"
    LC_ALL=C sort -zu "$selected_file" > "$sorted_file"
    mapfile -d '' -t selected_notebooks < "$sorted_file"
  fi
}

print_list() {
  local title="$1"
  shift
  local items=("$@")
  local item

  echo "$title (${#items[@]}):"
  if [[ "${#items[@]}" -eq 0 ]]; then
    echo "  none"
    return 0
  fi

  for item in "${items[@]}"; do
    echo "  $item"
  done
}

print_summary() {
  echo "::group::Notebook summary"
  echo "Selected: ${#selected_notebooks[@]}"
  echo "CI skipped: ${#ci_skipped_notebooks[@]}"
  echo "Passed: ${#passed_notebooks[@]}"
  echo "Failed: ${#failed_notebooks[@]}"
  echo "Timed out: ${#timed_out_notebooks[@]}"

  if [[ "${#ci_skipped_notebooks[@]}" -gt 0 ]]; then
    print_list "CI-skipped notebooks" "${ci_skipped_notebooks[@]}"
  fi

  if [[ "${#failed_notebooks[@]}" -gt 0 ]]; then
    print_list "Failed notebooks" "${failed_notebooks[@]}"
  fi

  if [[ "${#timed_out_notebooks[@]}" -gt 0 ]]; then
    print_list "Timed out notebooks" "${timed_out_notebooks[@]}"
  fi

  echo "::endgroup::"
}

notebook_log_file() {
  local notebook="$1"
  local log_name

  if [[ -n "${NOTEBOOK_LOG_DIR:-}" ]]; then
    log_name="${notebook//\//__}"
    printf '%s/%s.log\n' "$NOTEBOOK_LOG_DIR" "$log_name"
  else
    printf '%s/nbconvert.log\n' "$TMP_DIR"
  fi
}

run_notebook() {
  local notebook="$1"
  local output_file="$TMP_DIR/nbconvert-output.ipynb"
  local log_file
  local exit_code
  local start_seconds
  local duration_seconds

  log_file="$(notebook_log_file "$notebook")"

  rm -f "$log_file" "$output_file"

  echo "::group::Notebook: $notebook"
  echo "Running: $notebook"

  start_seconds=$SECONDS
  set +e
  if [[ "$EXEC_TIMEOUT_SECONDS" == "-1" ]]; then
    jupyter nbconvert \
      --to notebook \
      --execute "$notebook" \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      --ExecutePreprocessor.timeout=-1 \
      --output "$output_file" \
      >"$log_file" 2>&1
  else
    timeout "$EXEC_TIMEOUT_SECONDS" jupyter nbconvert \
      --to notebook \
      --execute "$notebook" \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      --ExecutePreprocessor.timeout="$EXEC_TIMEOUT_SECONDS" \
      --output "$output_file" \
      >"$log_file" 2>&1
  fi
  exit_code=$?
  set -e
  duration_seconds=$((SECONDS - start_seconds))
  rm -f "$output_file"

  if [[ "$exit_code" -eq 0 ]]; then
    echo "PASS: $notebook"
    echo "Duration: ${duration_seconds}s"
    passed_notebooks+=("$notebook")
  elif [[ "$exit_code" -eq 124 ]] ||
    grep -Eiq "CellTimeoutError|TimeoutError|timed out|Timeout waiting for execute reply|A cell timed out" "$log_file"; then
    echo "TIMEOUT: $notebook"
    echo "Duration: ${duration_seconds}s"
    echo "Full log: $log_file"
    echo "Last $NOTEBOOK_LOG_TAIL_LINES log lines:"
    tail -n "$NOTEBOOK_LOG_TAIL_LINES" "$log_file" || true
    timed_out_notebooks+=("$notebook")
  else
    echo "FAIL: $notebook"
    echo "Exit code: $exit_code"
    echo "Duration: ${duration_seconds}s"
    echo "Full log: $log_file"
    echo "Last $NOTEBOOK_LOG_TAIL_LINES log lines:"
    tail -n "$NOTEBOOK_LOG_TAIL_LINES" "$log_file" || true
    failed_notebooks+=("$notebook")
  fi

  echo "::endgroup::"
}

load_ci_skip_list

list_mode="full"
if [[ -n "$NOTEBOOK_LIST_FILE" ]]; then
  list_mode="explicit"
  read_explicit_list
else
  discover_full_suite
fi

sort_selected_notebooks

echo "::group::Notebook selection"
echo "Working directory: $(pwd)"
echo "Selection mode: $list_mode"

if [[ "${#ci_skip_notebooks[@]}" -gt 0 ]]; then
  echo "CI notebook exceptions are enabled."
  echo "Reason: ${NOTEBOOK_CI_SKIP_REASON:-these notebooks are explicit CI exceptions documented in the workflow.}"
  print_list "Configured CI notebook exceptions" "${ci_skip_notebooks[@]}"
fi

if [[ "${#excluded_notebooks[@]}" -gt 0 ]]; then
  print_list "Excluded notebook candidates" "${excluded_notebooks[@]}"
fi

if [[ "${#ci_skipped_notebooks[@]}" -gt 0 ]]; then
  print_list "CI-skipped notebook candidates" "${ci_skipped_notebooks[@]}"
fi

if [[ "${#missing_notebooks[@]}" -gt 0 ]]; then
  print_list "Missing notebook candidates" "${missing_notebooks[@]}"
  echo "::endgroup::"
  echo "ERROR: one or more selected notebook candidates do not exist." >&2
  exit 2
fi

print_list "Selected notebooks" "${selected_notebooks[@]}"
echo "::endgroup::"

if [[ "${#selected_notebooks[@]}" -eq 0 ]]; then
  if [[ "$list_mode" == "explicit" ]]; then
    echo "No notebooks selected after filtering the explicit notebook list; exiting successfully."
    print_summary
    exit 0
  fi

  echo "ERROR: full-suite discovery found no notebooks after filtering." >&2
  print_summary
  exit 2
fi

echo "::group::Notebook execution assumptions"
echo "Selected notebooks are executed as-is."
if [[ "${#ci_skip_notebooks[@]}" -gt 0 ]]; then
  echo "Only notebooks listed in NOTEBOOK_CI_SKIP_NOTEBOOKS are skipped."
  echo "CI exception reason: ${NOTEBOOK_CI_SKIP_REASON:-these notebooks are explicit CI exceptions documented in the workflow.}"
else
  echo "No notebooks are skipped for data files, checkpoints, GPU availability, credentials, or network access."
fi
echo "If a selected notebook requires unavailable resources, that notebook fails this run."
echo "::endgroup::"

for notebook in "${selected_notebooks[@]}"; do
  run_notebook "$notebook"
done

print_summary

if [[ "${#failed_notebooks[@]}" -gt 0 ]]; then
  exit 1
fi

if [[ "${#timed_out_notebooks[@]}" -gt 0 ]]; then
  exit 124
fi

exit 0

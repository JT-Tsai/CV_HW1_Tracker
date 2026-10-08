#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0

# Fill in the Google Drive file IDs before running this script.
export YOLO_MODEL_ID="1k1QSn-D61GbTLcbfQuEhXl1jGBz_MX_C"
export TEST_VIDEO_ID="1UVG5QPYWwQssrN4IRna9EeFOPjlAHSSX"
export TEST2_VIDEO_ID="1ElMITZW4ragfj2Wcj3fzp1Vzadqeb0jI"

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is required. Install it with: python -m pip install gdown" >&2
    exit 1
fi

: "${YOLO_MODEL_ID:?Edit scripts/download_assets.sh and set YOLO_MODEL_ID}"
: "${TEST_VIDEO_ID:?Edit scripts/download_assets.sh and set TEST_VIDEO_ID}"
: "${TEST2_VIDEO_ID:?Edit scripts/download_assets.sh and set TEST2_VIDEO_ID}"

GDOWN_ARGS=()
if [[ -n "${GDOWN_COOKIES_FILE:-}" ]]; then
    GDOWN_ARGS+=(--cookies "$GDOWN_COOKIES_FILE")
fi

download_asset() {
    local file_id="$1"
    local destination="$2"
    local output_path="$ROOT_DIR/$destination"

    if [[ -f "$output_path" && "${FORCE_DOWNLOAD:-0}" != "1" ]]; then
        echo "Exists: $destination"
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "Would download: $destination"
        return
    fi

    mkdir -p "$(dirname "$output_path")"
    echo "Downloading: $destination"
    gdown "${GDOWN_ARGS[@]}" --id "$file_id" --output "$output_path"
}

download_asset "$YOLO_MODEL_ID" "models/yolo/best.pt"
download_asset "$TEST_VIDEO_ID" "assets/videos/test.mp4"
download_asset "$TEST2_VIDEO_ID" "assets/videos/test2.mp4"

echo "Asset download complete."

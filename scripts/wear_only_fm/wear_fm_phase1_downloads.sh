#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned}"
cd "$PROJECT_ROOT"

BASE="outputs/wear_fm"
LOG_DIR="$BASE/logs"
PID_FILE="$LOG_DIR/phase1_download.pid"
LOG_FILE="$LOG_DIR/phase1_download.log"
STATUS_FILE="$LOG_DIR/download_status_latest.log"

mkdir -p \
  "$LOG_DIR" \
  "$BASE/third_party/ssl-wearables/sslearning/models" \
  "$BASE/third_party/ssl-wearables/model_check_point" \
  "$BASE/weights"

write_status() {
  {
    echo "== checked $(date -Is) =="
    echo
    echo "== disk =="
    df -h /vePFS-0x0d /tmp
    echo
    echo "== download processes =="
    pgrep -af 'curl|wget|aria2c|phase1_download' || true
    echo
    echo "== files =="
    find "$BASE/third_party" "$BASE/weights" -maxdepth 4 -type f -printf '%s %p\n' 2>/dev/null | sort -nr | head -40 || true
    echo
    if [[ -f "$LOG_FILE" ]]; then
      echo "== log tail =="
      tail -40 "$LOG_FILE"
    fi
  } > "$STATUS_FILE"
  cat "$STATUS_FILE"
}

case "${1:-start}" in
  status)
    write_status
    ;;
  start)
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "download already running pid=$(cat "$PID_FILE")"
      write_status
      exit 0
    fi

    nohup bash -lc '
      set -u
      cd "$0"
      BASE="outputs/wear_fm"
      LOG="$BASE/logs/phase1_download.log"

      fetch_checked() {
        url="$1"
        out="$2"
        expected="$3"
        part="${out}.part"
        echo "--- $(date -Is) downloading $out"
        rm -f "$out"
        wget -c --tries=20 --timeout=30 --waitretry=10 -O "$part" "$url"
        actual=$(stat -c %s "$part")
        if [ "$actual" != "$expected" ]; then
          echo "size mismatch for $out: expected=$expected actual=$actual" >&2
          exit 1
        fi
        mv "$part" "$out"
        sha256sum "$out"
        ls -lh "$out"
      }

      {
        echo "started $(date -Is)"
        df -h /vePFS-0x0d /tmp
        fetch_checked "https://codeload.github.com/nokia-bell-labs/papagei-foundation-model/zip/refs/heads/main" "$BASE/third_party/papagei-main.zip" "878444"
        fetch_checked "https://zenodo.org/records/13983110/files/papagei_s.pt?download=1" "$BASE/weights/papagei_s.pt" "23339216"
        fetch_checked "https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/hubconf.py" "$BASE/third_party/ssl-wearables/hubconf.py" "4506"
        fetch_checked "https://raw.githubusercontent.com/OxWearables/ssl-wearables/main/sslearning/models/accNet.py" "$BASE/third_party/ssl-wearables/sslearning/models/accNet.py" "26865"
        touch "$BASE/third_party/ssl-wearables/sslearning/__init__.py" "$BASE/third_party/ssl-wearables/sslearning/models/__init__.py"
        fetch_checked "https://wearables-files.ndph.ox.ac.uk/files/ssl/mtl_best.mdl" "$BASE/third_party/ssl-wearables/model_check_point/mtl_best.mdl" "42014098"
        fetch_checked "https://codeload.github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/zip/refs/heads/main" "$BASE/third_party/normwear-main.zip" "1319590"
        fetch_checked "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/download/v1.0.0-alpha/normwear_pretrain_ckpt.pth" "$BASE/weights/normwear_pretrain_ckpt.pth" "544579503"
        echo "finished $(date -Is)"
        df -h /vePFS-0x0d /tmp
      } >> "$LOG" 2>&1
    ' "$PROJECT_ROOT" >/dev/null 2>&1 &

    echo $! > "$PID_FILE"
    echo "started download pid=$(cat "$PID_FILE")"
    write_status
    ;;
  *)
    echo "usage: $0 [start|status]" >&2
    exit 2
    ;;
esac

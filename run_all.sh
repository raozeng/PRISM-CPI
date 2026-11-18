#!/usr/bin/env bash
# ==============================================================================
# PRISM-CPI Experiment Runner & Background Process Manager
#
# Usage:
#   bash run_all.sh start       (Run in background via nohup with checkpoint resume)
#   bash run_all.sh             (Run in foreground with checkpoint resume)
#   bash run_all.sh status      (Show active process, GPU status, progress & ETA)
#   bash run_all.sh logs        (Stream training logs via tail -f)
#   bash run_all.sh verify      (Check completed experiment outputs)
#   bash run_all.sh package     (Bundle results/ and checkpoints/best*.pth)
#   bash run_all.sh stop        (Stop active background run)
#   bash run_all.sh dry-run     (Quick functional test on synthetic data)
#   bash run_all.sh rerun       (Re-run experiments from scratch)
# ==============================================================================
set -euo pipefail

# Ignore SIGHUP so closing SSH terminal never kills foreground or child processes
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="results/logs"
PID_FILE="${LOG_DIR}/runner.pid"
MASTER_LOG="${LOG_DIR}/master_run.log"
mkdir -p data checkpoints "${LOG_DIR}" results/tables results/runs

ACTION="run"
MODE="full"
FORCE_RERUN_FLAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        start|daemon|bg|--daemon|-d|--bg)
            ACTION="start"
            shift
            ;;
        status|--status)
            ACTION="status"
            shift
            ;;
        logs|log|--logs)
            ACTION="logs"
            shift
            ;;
        verify|--verify)
            ACTION="verify"
            shift
            ;;
        package|pack|--package)
            ACTION="package"
            shift
            ;;
        stop|--stop)
            ACTION="stop"
            shift
            ;;
        dry-run|dryrun|--dry-run)
            MODE="dry-run"
            EXTRA_ARGS="${EXTRA_ARGS} --dry-run"
            shift
            ;;
        rerun|force-rerun|restart|--force-rerun)
            FORCE_RERUN_FLAG="--force-rerun"
            EXTRA_ARGS="${EXTRA_ARGS} --force-rerun"
            shift
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

if [[ "${ACTION}" == "start" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
        OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
            echo "[INFO] Experiment runner is ALREADY running in background (PID: ${OLD_PID})."
            echo "       Use 'bash run_all.sh status' to inspect progress or 'bash run_all.sh logs' to view logs."
            exit 0
        fi
    fi
    echo "============================================================"
    echo "[SSH-Safe Background Runner] Launching PRISM-CPI experiment pipeline..."
    nohup bash "${BASH_SOURCE[0]}" ${EXTRA_ARGS} >> "${MASTER_LOG}" 2>&1 &
    BG_PID=$!
    disown "${BG_PID}" 2>/dev/null || true
    echo "${BG_PID}" > "${PID_FILE}"
    echo "  -> Background PID : ${BG_PID} (saved to ${PID_FILE})"
    echo "  -> Master Log     : ${MASTER_LOG}"
    echo "  -> Check Status   : bash run_all.sh status"
    echo "  -> Follow Logs    : bash run_all.sh logs"
    echo "  -> Verify Anytime : bash run_all.sh verify"
    echo "  -> Package Result : bash run_all.sh package"
    echo "  -> Stop Runner    : bash run_all.sh stop"
    echo "You can now safely disconnect from SSH!"
    echo "============================================================"
    exit 0
fi

if [[ "${ACTION}" == "stop" ]]; then
    if [[ -f "${PID_FILE}" ]]; then
        TARGET_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${TARGET_PID}" ]] && kill -0 "${TARGET_PID}" 2>/dev/null; then
            echo "[STOP] Stopping background experiment runner (PID: ${TARGET_PID})..."
            kill -TERM "${TARGET_PID}" 2>/dev/null || true
            echo "[STOP] Signal sent. Completed experiments and epoch checkpoints are safely preserved for auto-resume."
        else
            echo "[INFO] No active process found for PID ${TARGET_PID}."
        fi
        rm -f "${PID_FILE}"
    else
        echo "[INFO] No PID file found at ${PID_FILE}."
    fi
    exit 0
fi

if [[ "${ACTION}" == "status" ]]; then
    echo "============================================================"
    echo "[SCI Experiment Runner - Live Status Monitor]"
    if [[ -f "${PID_FILE}" ]]; then
        CUR_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [[ -n "${CUR_PID}" ]] && kill -0 "${CUR_PID}" 2>/dev/null; then
            echo "Background Process : RUNNING (PID: ${CUR_PID})"
        else
            echo "Background Process : NOT RUNNING (Last PID: ${CUR_PID})"
        fi
    else
        echo "Background Process : NO ACTIVE BACKGROUND PID FILE"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "------------------------------------------------------------"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
    echo "------------------------------------------------------------"
    python3 run.py --status ${EXTRA_ARGS}
    exit 0
fi

if [[ "${ACTION}" == "logs" ]]; then
    echo "[Logs] Streaming ${MASTER_LOG} (Press Ctrl+C to exit log viewer; training will keep running)..."
    touch "${MASTER_LOG}"
    tail -n 60 -f "${MASTER_LOG}"
    exit 0
fi

if [[ "${ACTION}" == "verify" ]]; then
    python3 test.py --mode "${MODE}" --verify
    exit 0
fi

if [[ "${ACTION}" == "package" ]]; then
    python3 test.py --mode "${MODE}" --aggregate --verify --package
    exit 0
fi

echo "============================================================"
echo "[SCI Experiment Runner] Starting PRISM-CPI in mode: ${MODE} (Auto-Resume Enabled)"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "============================================================"

if [[ "${MODE}" == "dry-run" ]]; then
    DRY_DATA_DIR="./data_dry_run"
    echo "[Step 1/4] Preparing synthetic smoke-test datasets in isolated ${DRY_DATA_DIR} (preserving ./data/processed/ full datasets)..."
    python3 prepare_data.py --mode dry-run --data-dir "${DRY_DATA_DIR}" ${FORCE_RERUN_FLAG}

    echo "[Step 2/4] Running SOTA Baselines, Proposed Model & Ablations via run.py (dry-run)..."
    python3 run.py --mode train --exp_group all --data-dir "${DRY_DATA_DIR}" ${EXTRA_ARGS}

    echo "[Step 3/4] Running evaluation, completeness verification & best.pth packaging (test.py)..."
    python3 test.py --mode dry-run --data-dir "${DRY_DATA_DIR}" --aggregate --verify --package --copy-dry-run-to ../dry_run_output/metrics_summary.json
    cp -f results/integrity_report.json ../dry_run_output/integrity_report.json 2>/dev/null || true
    rm -rf "${DRY_DATA_DIR}"
else
    echo "[Step 1/4] Verifying/loading locally pre-processed full datasets in ./data/processed/ (zero truncation)..."
    python3 prepare_data.py --mode full ${FORCE_RERUN_FLAG}

    echo "[Step 2/4] Running SOTA Baselines, Proposed Model & Ablations via run.py (with Auto-Resume & best.pth saving)..."
    python3 run.py --mode train --exp_group all ${EXTRA_ARGS}

    echo "[Step 3/4] Running independent evaluation, multi-seed aggregation, completeness verification & best.pth packaging (test.py)..."
    python3 test.py --mode full --aggregate --verify --package
fi

echo "============================================================"
echo "[Step 4/4] Pipeline execution finished (mode=${MODE})."
echo "============================================================"

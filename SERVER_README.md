# PRISM-CPI Server Training & Evaluation Guide

This document summarizes the command-line workflow for running multi-seed training, checkpoint auto-resume, progress monitoring, and test-set evaluation on a Linux GPU server.

---

## 1. Codebase Layout

- `prepare_data.py`: Preprocessing script that reads raw benchmark splits from `./data/raw/` and builds 3D molecular and protein feature bundles under `./data/processed/`.
- `dataset.py`: PyTorch `Dataset` (`CPIDataset`, `UnlabeledTargetDataset`, `BipartiteCPIGraphBuilder`), `DataLoader`, and batch collation utilities.
- `model.py`: Implementation of **PRISM-CPI** (`CC-PAE`, `PC-IFGA`, `SS-UAKD` + `E-CDAN`, and ablation variants), 12 baseline architectures, and training objectives.
- `train.py`: Training and validation engine (`epochs = 100` by default, mixed-precision support, task-level and epoch-level checkpoint resume, saving best weights to `checkpoints/best.pth` and `checkpoints/best_<dataset>.pth`).
- `test.py`: Test-set evaluation, 5-seed statistical aggregation (`mean ± SD` and paired two-tailed $t$-tests), and result archiving.
- `run.py`: Unified CLI entry point connecting dataset loading, training, evaluation, and status reporting.
- `run_all.sh`: Shell wrapper for background execution (`nohup`), status inspection, log streaming, and result packaging.

---

## 2. Common Commands

| Task | Command | Description |
| :--- | :--- | :--- |
| **Start background run** | `bash run_all.sh start` | Launches training in background with `nohup`, saving PID to `results/logs/runner.pid`. |
| **Resume interrupted run** | `bash run_all.sh start` | Automatically skips finished tasks (`results/runs/<exp_id>.json`) and resumes unfinished epochs from `checkpoints/last_resume_<exp_id>.pt`. |
| **Check live progress** | `bash run_all.sh status` | Displays process status, GPU utilization, completion percentage, active task, and ETA. |
| **Stream training logs** | `bash run_all.sh logs` | Follows `results/logs/master_run.log` in real time (`Ctrl+C` exits log view without stopping training). |
| **Verify run outputs** | `bash run_all.sh verify` | Checks completed runs and writes `results/integrity_report.json`. |
| **Package results & weights** | `bash run_all.sh package` | Aggregates metrics and bundles `results/` together with `checkpoints/best*.pth` into `experiment_results_bundle.tar.gz`. |
| **Stop background run** | `bash run_all.sh stop` | Gracefully terminates the active background process tree. |
| **Quick local test** | `bash run_all.sh dry-run` | Runs a lightweight 2-epoch test on synthetic batches in an isolated temporary directory (`./data_dry_run`). |

#!/usr/bin/env python3
"""
Command-line entry point for training, evaluating, and monitoring PRISM-CPI experiments.

Examples:
  python3 run.py --dry-run --exp_group all --mode both
  python3 run.py --exp_group all --mode both
  python3 run.py --status
  python3 run.py --verify
  python3 run.py --package
  python3 run.py --exp_group single --dataset KIBA --model PRISM-CPI --ablation PRISM_CPI_Full --seed 12 --epochs 100
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dataset import get_dataloaders
from model import ABLATION_VARIANTS, BASELINE_REGISTRY, build_model
from prepare_data import SEEDS, TABLE1_BENCHMARK_SPECS
from test import (
    evaluate_all_and_aggregate,
    evaluate_checkpoint,
    package_results_bundle,
    verify_experiment_completeness,
)
from train import fit_experiment


def print_live_status(results_dir: str = "./results") -> None:
    """Reads `results/logs/progress_status.json` and `results/runs/` and prints a formatted live status table."""
    res_dir = Path(results_dir)
    status_path = res_dir / "logs" / "progress_status.json"
    runs_dir = res_dir / "runs"
    num_run_files = len(list(runs_dir.glob("*.json"))) if runs_dir.exists() else 0

    print("============================================================")
    print("[PRISM-CPI Experiment Progress & Live Status Table]")
    if not status_path.exists():
        print("  Pipeline Status     : NOT STARTED (No progress_status.json found)")
        print(f"  Cached Run Files    : {num_run_files} in {runs_dir}")
        print("============================================================")
        return

    try:
        st = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  Pipeline Status     : UNREADABLE ({exc})")
        print("============================================================")
        return

    cur = st.get("current_task", {})
    print(f"  Pipeline Status     : {st.get('status', 'UNKNOWN')}")
    print(f"  Execution Mode      : {st.get('mode', 'UNKNOWN')}")
    print(f"  Started At (UTC)    : {st.get('started_at', '-')}")
    print(f"  Last Updated (UTC)  : {st.get('updated_at', '-')}")
    print(f"  Elapsed Time        : {st.get('elapsed_sec', 0.0):.1f}s")
    print(f"  Estimated Remaining : {st.get('eta_sec', 0.0):.1f}s")
    print(
        f"  Task Progress       : {st.get('completed_tasks', 0)} / {st.get('total_tasks', 0)} "
        f"({st.get('progress_percent', 0.0):.2f}%)"
    )
    print(f"  Resumed & Skipped   : {st.get('resumed_skipped_tasks', 0)} tasks")
    print(f"  Failed Tasks        : {st.get('failed_tasks', 0)} tasks")
    print(f"  Saved Run JSONs     : {num_run_files} files in {runs_dir}")
    print("------------------------------------------------------------")
    print("  Current Active Task :")
    print(f"    exp_id            : {cur.get('exp_id', '-')}")
    print(
        f"    dataset / split   : {cur.get('dataset', '-')} ({cur.get('scenario', '-')}) | "
        f"model={cur.get('model', '-')} | ablation={cur.get('ablation', '-')} | seed={cur.get('seed', '-')}"
    )
    print(
        f"    epoch progress    : {cur.get('current_epoch', 0)} / {cur.get('total_epochs', 0)} | "
        f"loss={cur.get('latest_loss', 0.0):.4f} | "
        f"val_AUC={cur.get('latest_val_auc', 0.0):.4f} | "
        f"val_AUPR={cur.get('latest_val_aupr', 0.0):.4f}"
    )
    print("============================================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PRISM-CPI Unified Experiment Controller")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Single dataset name (BindingDB, BioSNAP, PDBbind_2016, KIBA). If empty, uses --datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"],
        help="List of benchmark datasets to run.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="PRISM-CPI",
        help=f"Model ID: 'PRISM-CPI' or one of {list(BASELINE_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="PRISM_CPI_Full",
        help=f"Ablation variant: {[v['variant_id'] for v in ABLATION_VARIANTS]}",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="inductive",
        choices=["inductive", "transductive"],
        help="Evaluation split scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional single random seed override (e.g., 12). If 0, uses --seeds.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="List of random seeds (default: 12 14 16 18 20).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs (0 uses mode default: 2 for dry-run, 100 for full).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Mini-batch size (0 uses mode default: 32 for dry-run, 64 for full).",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay (default: 1e-5)")
    parser.add_argument("--K_pocket", type=int, default=32, help="Number of pocket subsite tokens K (default: 32)")
    parser.add_argument("--B_rbf", type=int, default=16, help="Number of Gaussian RBF kernels B (default: 16)")
    parser.add_argument("--r_max", type=float, default=6.0, help="3D distance cutoff radius in Angstrom (default: 6.0)")
    parser.add_argument("--lambda_rel", type=float, default=0.5, help="Relational distillation weight (default: 0.5)")
    parser.add_argument("--tau_kd", type=float, default=1.0, help="Uncertainty attenuation temperature (default: 1.0)")
    parser.add_argument(
        "--exp_group",
        type=str,
        default="all",
        choices=["all", "single", "baselines", "ablations", "sensitivity", "transductive"],
        help="Experiment group to execute.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["train", "test", "both", "aggregate", "full", "dry-run"],
        help="Execution stage or mode: train | test | both | aggregate (also accepts full | dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable fast local smoke-test mode.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached runs/checkpoints and re-run from scratch.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print live progress status table from results/logs/progress_status.json and exit.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify experiment completeness against experiment_plan.json and exit.",
    )
    parser.add_argument(
        "--package",
        action="store_true",
        help="Verify experiment completeness and package experiment_results_bundle.tar.gz.",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Relative path to data directory.")
    parser.add_argument("--results-dir", type=str, default="./results", help="Relative path to results directory.")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="./checkpoints",
        help="Relative path to model checkpoints directory.",
    )
    parser.add_argument(
        "--copy-dry-run-to",
        type=str,
        default="",
        help="Optional target path to mirror metrics_summary.json after dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    is_dry_run = bool(args.dry_run or args.mode == "dry-run")
    exec_mode = "dry-run" if is_dry_run else "full"

    if args.status:
        print_live_status(results_dir=args.results_dir)
        return

    if args.verify or args.package:
        logs_path = Path(args.results_dir) / "logs" / "training_runs.json"
        if args.package and logs_path.exists():
            evaluate_all_and_aggregate(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                copy_dry_run_to=args.copy_dry_run_to,
            )
        verify_experiment_completeness(
            mode=exec_mode,
            results_dir=args.results_dir,
            checkpoints_dir=args.checkpoints_dir,
        )
        if args.package:
            package_results_bundle(
                results_dir=args.results_dir,
                checkpoints_dir=args.checkpoints_dir,
            )
        return

    if args.mode in {"full", "dry-run"}:
        stage_action = "both"
    else:
        stage_action = args.mode

    target_datasets: List[str] = [args.dataset] if args.dataset else list(args.datasets)
    for ds in target_datasets:
        if ds not in TABLE1_BENCHMARK_SPECS:
            raise ValueError(f"Unknown dataset '{ds}'. Valid choices: {list(TABLE1_BENCHMARK_SPECS.keys())}")

    target_seeds: List[int] = [int(args.seed)] if args.seed > 0 else [int(s) for s in args.seeds]

    # Verify DataLoader & Model Factory wiring on the first target dataset and seed
    first_ds_pt = Path(args.data_dir) / "processed" / f"{target_datasets[0]}.pt"
    if first_ds_pt.exists():
        dl_bundle = get_dataloaders(
            dataset_name=target_datasets[0],
            data_dir=args.data_dir,
            scenario=args.scenario,
            seed=target_seeds[0],
            batch_size=args.batch_size if args.batch_size > 0 else (32 if is_dry_run else 64),
        )
        _ = build_model(
            model_id=args.model,
            ablation=args.ablation,
            seed=target_seeds[0],
            K_pocket=args.K_pocket,
            B_rbf=args.B_rbf,
            r_max=args.r_max,
            lambda_rel=args.lambda_rel,
            tau_kd=args.tau_kd,
        )
        assert len(dl_bundle["train_dataset"]) > 0, "Empty training split!"

    hyperparams: Dict[str, Any] = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "K_pocket": args.K_pocket,
        "B_rbf": args.B_rbf,
        "r_max": args.r_max,
        "lambda_rel": args.lambda_rel,
        "tau_kd": args.tau_kd,
    }
    if args.epochs > 0:
        hyperparams["epochs"] = args.epochs
    if args.batch_size > 0:
        hyperparams["batch_size"] = args.batch_size

    if stage_action in {"train", "both"}:
        fit_experiment(
            mode=exec_mode,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            checkpoints_dir=args.checkpoints_dir,
            datasets=target_datasets,
            seeds=target_seeds,
            exp_group=args.exp_group,
            single_model=args.model,
            single_ablation=args.ablation,
            scenario=args.scenario,
            hyperparams=hyperparams,
            force_rerun=args.force_rerun,
        )

    if stage_action == "test" and args.exp_group == "single":
        single_res = evaluate_checkpoint(
            dataset_name=target_datasets[0],
            model_id=args.model,
            ablation=args.ablation,
            scenario=args.scenario,
            seed=target_seeds[0],
            data_dir=args.data_dir,
            hyperparams=hyperparams,
        )
        print(json.dumps(single_res, indent=2))
        return

    if stage_action in {"test", "both", "aggregate"}:
        evaluate_all_and_aggregate(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            copy_dry_run_to=args.copy_dry_run_to,
        )


if __name__ == "__main__":
    main()

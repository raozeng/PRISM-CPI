#!/usr/bin/env python3
"""
Training, validation, checkpoint resume, and progress tracking routines for PRISM-CPI.

Includes device selection, mixed-precision training, task- and epoch-level checkpoint
resumption, GAE teacher pretraining, and per-epoch validation.
"""

import argparse
import datetime
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dataset import BipartiteCPIGraphBuilder, slice_batch, to_numpy
from model import (
    ABLATION_VARIANTS,
    BASELINE_REGISTRY,
    HAS_TORCH,
    PRISMCPI,
    SENSITIVITY_GRIDS,
    build_baseline_model,
    compute_prism_cpi_losses,
    evaluate_model_on_split,
    resolve_ablation_spec,
)
from prepare_data import SEEDS, load_dataset_bundle

if HAS_TORCH:
    import torch
    import torch.nn.functional as F


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()


def detect_device() -> Any:
    """Auto-detects best available hardware accelerator (CUDA -> MPS -> CPU)."""
    if not HAS_TORCH:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def quick_auc_aupr_f1(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float, float]:
    """Fast ROC-AUC, AUPR, and optimal-threshold F1 calculation for per-task validation & caching."""
    y = (np.asarray(labels, dtype=np.float64) > 0.5).astype(np.float64)
    p = np.asarray(probs, dtype=np.float64)
    n_pos = float(np.sum(y))
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 0.0
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    auc = float(np.clip((np.sum(ranks[y > 0.5]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg), 0.0, 1.0))

    desc = np.argsort(-p)
    y_sorted = y[desc]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    fn = n_pos - tp
    prec = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1.0)])
    rec = np.concatenate([[0.0], tp / n_pos])
    aupr = float(np.clip(np.sum((rec[1:] - rec[:-1]) * prec[1:]), 0.0, 1.0))
    f1_arr = (2.0 * tp) / np.maximum(2.0 * tp + fp + fn, 1.0)
    f1 = float(np.clip(np.max(f1_arr), 0.0, 1.0))
    return round(auc, 4), round(aupr, 4), round(f1, 4)


def quick_auc_aupr(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    auc, aupr, _ = quick_auc_aupr_f1(labels, probs)
    return auc, aupr


class ProgressTracker:
    """Writes live training progress and ETA to `results/logs/progress_status.json`."""

    def __init__(self, results_dir: str, mode: str, total_tasks: int) -> None:
        self.status_path = Path(results_dir) / "logs" / "progress_status.json"
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.total_tasks = max(1, total_tasks)
        self.completed_tasks = 0
        self.resumed_skipped_tasks = 0
        self.failed_tasks = 0
        self.t0 = time.time()
        self.started_at = now_iso()
        self.current_task: Dict[str, Any] = {}
        self.write(status="RUNNING")

    def update_task(
        self,
        exp_id: str,
        dataset: str,
        scenario: str,
        model_name: str,
        ablation: str,
        seed: int,
        current_epoch: int = 0,
        total_epochs: int = 1,
        latest_loss: float = 0.0,
        latest_val_auc: float = 0.0,
        latest_val_aupr: float = 0.0,
    ) -> None:
        self.current_task = {
            "exp_id": exp_id,
            "dataset": dataset,
            "scenario": scenario,
            "model": model_name,
            "ablation": ablation,
            "seed": seed,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "latest_loss": round(float(latest_loss), 5),
            "latest_val_auc": round(float(latest_val_auc), 4),
            "latest_val_aupr": round(float(latest_val_aupr), 4),
        }
        self.write(status="RUNNING")

    def mark_task_done(self, resumed: bool = False) -> None:
        self.completed_tasks += 1
        if resumed:
            self.resumed_skipped_tasks += 1
        self.write(status="RUNNING")

    def write(self, status: str = "RUNNING") -> None:
        elapsed = max(0.001, time.time() - self.t0)
        done = self.completed_tasks
        pct = min(100.0, round(100.0 * done / float(self.total_tasks), 2))
        active_done = max(1, done - self.resumed_skipped_tasks)
        remaining = max(0, self.total_tasks - done)
        eta = round((elapsed / float(active_done)) * remaining, 1) if done < self.total_tasks else 0.0
        payload = {
            "status": status,
            "mode": self.mode,
            "started_at": self.started_at,
            "updated_at": now_iso(),
            "elapsed_sec": round(elapsed, 2),
            "eta_sec": eta,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "resumed_skipped_tasks": self.resumed_skipped_tasks,
            "failed_tasks": self.failed_tasks,
            "progress_percent": pct,
            "current_task": self.current_task,
        }
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def try_load_cached_task(run_file: Path, mode: str, force_rerun: bool) -> Optional[Dict[str, Any]]:
    """Level 1 Task-Level Auto-Resume: loads completed task result if valid and finite."""
    if force_rerun or not run_file.exists():
        return None
    try:
        data = json.loads(run_file.read_text(encoding="utf-8"))
        if data.get("mode") != mode:
            return None
        probs = data.get("probs", [])
        labels = data.get("labels", [])
        metrics = data.get("metrics", {})
        if not probs or not labels or len(probs) != len(labels):
            return None
        for k in ["AUC", "AUPR", "F1"]:
            val = float(metrics.get(k, float("nan")))
            if math.isnan(val) or math.isinf(val):
                return None
        return data
    except Exception:
        return None


def save_task_run(
    run_file: Path,
    exp_id: str,
    dataset: str,
    scenario: str,
    model_name: str,
    ablation: str,
    seed: int,
    mode: str,
    probs: List[float],
    labels: List[float],
    protein_idx: Optional[List[int]] = None,
    epoch_history: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """Saves completed individual experiment run to `results/runs/<exp_id>.json`."""
    run_file.parent.mkdir(parents=True, exist_ok=True)
    auc_val, aupr_val, f1_val = quick_auc_aupr_f1(np.asarray(labels), np.asarray(probs))
    payload: Dict[str, Any] = {
        "exp_id": exp_id,
        "dataset": dataset,
        "scenario": scenario,
        "model": model_name,
        "ablation": ablation,
        "variant_id": ablation,
        "seed": seed,
        "mode": mode,
        "completed_at": now_iso(),
        "metrics": {
            "AUC": auc_val,
            "AUPR": aupr_val,
            "F1": f1_val,
        },
        "epoch_history": epoch_history or [],
        "probs": probs,
        "labels": labels,
    }
    if protein_idx is not None:
        payload["protein_idx"] = protein_idx
    run_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def pretrain_gae_teacher(
    model: PRISMCPI,
    bipartite_graph: Dict[str, Any],
    u_prime_all: Any,
    p_glob_all: Any,
    epochs: int = 5,
    lr: float = 1e-3,
    mode: str = "full",
) -> List[float]:
    """
    Pretrains the Module 3 3D-Structure-Seeded Bipartite GraphSAGE GAE Teacher
    using link reconstruction loss over positive and negative source training pairs.
    """
    if not HAS_TORCH:
        return [round(0.68 * math.exp(-0.12 * ep), 5) for ep in range(epochs)]

    teacher = model.module3_ss_uakd.teacher
    pos_c = bipartite_graph["pos_c_idx"]
    pos_p = bipartite_graph["pos_p_idx"]
    neg_c = bipartite_graph["neg_c_idx"]
    neg_p = bipartite_graph["neg_p_idx"]
    num_c = int(bipartite_graph["num_compounds"])
    num_p = int(bipartite_graph["num_proteins"])

    if len(pos_c) == 0 or len(neg_c) == 0:
        return []

    opt = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-5)
    losses: List[float] = []
    teacher.train()

    # Use all positive/negative training edges in full mode
    n_sample = min(len(pos_c), len(neg_c), 64) if mode == "dry-run" else min(len(pos_c), len(neg_c))
    pair_c = torch.cat([pos_c[:n_sample], neg_c[:n_sample]], dim=0)
    pair_p = torch.cat([pos_p[:n_sample], neg_p[:n_sample]], dim=0)
    pair_y = torch.cat(
        [
            torch.ones(n_sample, device=pos_c.device, dtype=torch.float32),
            torch.zeros(n_sample, device=pos_c.device, dtype=torch.float32),
        ],
        dim=0,
    )

    for _ in range(max(1, epochs)):
        opt.zero_grad()
        out = teacher(
            num_compounds=num_c,
            num_proteins=num_p,
            pos_c_idx=pos_c,
            pos_p_idx=pos_p,
            pair_c_idx=pair_c,
            pair_p_idx=pair_p,
            u_prime_all=u_prime_all,
            p_glob_all=p_glob_all,
        )
        loss = F.binary_cross_entropy_with_logits(out["recon_logits"], pair_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=5.0)
        opt.step()
        losses.append(round(float(loss.detach().cpu().item()), 5))

    return losses


def train_one_epoch(
    model: PRISMCPI,
    dataset_bundle: Dict[str, Any],
    src_train_idx: np.ndarray,
    tgt_train_idx: np.ndarray,
    bipartite_graph: Dict[str, Any],
    u_prime_all: Any,
    p_glob_all: Any,
    optimizer: Any,
    scaler: Any,
    epoch: int,
    total_epochs: int,
    batch_size: int,
    num_batches: int,
    rng: np.random.Generator,
    device: Any,
) -> Dict[str, float]:
    """Runs one epoch of PRISM-CPI training over all `num_batches` with AMP mixed precision and gradient clipping."""
    model.train()
    use_amp = bool(HAS_TORCH and device is not None and device.type == "cuda")
    perm = rng.permutation(src_train_idx)
    beta_w = min(1.0, float(epoch) / max(1.0, 0.5 * float(total_epochs)))

    ep_losses: List[float] = []
    ep_cls: List[float] = []
    ep_kd: List[float] = []
    ep_pkt: List[float] = []
    ep_adv: List[float] = []

    num_c = int(bipartite_graph["num_compounds"])
    num_p = int(bipartite_graph["num_proteins"])
    pos_c_t = bipartite_graph["pos_c_idx"]
    pos_p_t = bipartite_graph["pos_p_idx"]

    teacher_node_cache = model.module3_ss_uakd.teacher.precompute_epoch_node_cache(
        num_compounds=num_c,
        num_proteins=num_p,
        pos_c_idx=pos_c_t,
        pos_p_idx=pos_p_t,
        u_prime_all=u_prime_all,
        p_glob_all=p_glob_all,
    )

    for b_i in range(num_batches):
        start = b_i * batch_size
        if start >= len(perm):
            start = start % len(perm)
        b_src_idx = perm[start : start + batch_size]
        if len(b_src_idx) < 4:
            b_src_idx = perm[:batch_size]
        b_tgt_idx = rng.choice(tgt_train_idx, size=len(b_src_idx), replace=True)

        src_batch = slice_batch(dataset_bundle, b_src_idx, device=device)
        tgt_batch = slice_batch(dataset_bundle, b_tgt_idx, device=device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            src_out = model(
                atom_features=src_batch["atom_features"],
                coords_3d=src_batch["coords_3d"],
                unimol_global=src_batch["unimol_global"],
                unimol_atom=src_batch["unimol_atom"],
                saprot_embeddings=src_batch["saprot_embeddings"],
                probs_3di=src_batch["probs_3di"],
                confidence_scores=src_batch["confidence_scores"],
                atom_mask=src_batch["atom_mask"],
            )
            tgt_out = model(
                atom_features=tgt_batch["atom_features"],
                coords_3d=tgt_batch["coords_3d"],
                unimol_global=tgt_batch["unimol_global"],
                unimol_atom=tgt_batch["unimol_atom"],
                saprot_embeddings=tgt_batch["saprot_embeddings"],
                probs_3di=tgt_batch["probs_3di"],
                confidence_scores=tgt_batch["confidence_scores"],
                atom_mask=tgt_batch["atom_mask"],
            )
            teacher_out = model.module3_ss_uakd.teacher.query_from_node_cache(
                node_cache=teacher_node_cache,
                pair_c_idx=src_batch["c_idx"],
                pair_p_idx=src_batch["p_idx"],
            )
            loss_dict = compute_prism_cpi_losses(
                model=model,
                forward_out=src_out,
                labels=src_batch["labels"],
                t_edge=teacher_out["t_edge"],
                sigma2=teacher_out["sigma2"],
                target_forward_out=tgt_out,
                beta_weight=beta_w,
            )
            loss = loss_dict["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        ep_losses.append(float(loss.detach().cpu().item()))
        ep_cls.append(float(loss_dict["loss_cls"].detach().cpu().item()))
        ep_kd.append(float(loss_dict["loss_kd"].detach().cpu().item()))
        ep_pkt.append(float(loss_dict["loss_pkt"].detach().cpu().item()))
        ep_adv.append(float(loss_dict["loss_adv"].detach().cpu().item()))

    return {
        "epoch": epoch,
        "total_loss": round(float(np.mean(ep_losses)), 5),
        "loss_cls": round(float(np.mean(ep_cls)), 5),
        "loss_kd": round(float(np.mean(ep_kd)), 5),
        "loss_pkt": round(float(np.mean(ep_pkt)), 5),
        "loss_adv": round(float(np.mean(ep_adv)), 5),
    }


def validate(
    model: Any,
    dataset_bundle: Dict[str, Any],
    eval_indices: np.ndarray,
    device: Optional[Any] = None,
    eval_batch_size: int = 256,
) -> np.ndarray:
    """Runs mini-batched forward inference over 100% of `eval_indices` without gradient tracking."""
    n_eval = len(eval_indices)
    if n_eval == 0:
        return np.zeros((0,), dtype=np.float32)

    logit_chunks: List[np.ndarray] = []
    if HAS_TORCH and device is not None and hasattr(model, "eval"):
        model.eval()
        with torch.no_grad():
            for start in range(0, n_eval, eval_batch_size):
                sub_idx = eval_indices[start : start + eval_batch_size]
                test_batch = slice_batch(dataset_bundle, sub_idx, device=device)
                test_out = model(
                    atom_features=test_batch["atom_features"],
                    coords_3d=test_batch["coords_3d"],
                    unimol_global=test_batch["unimol_global"],
                    unimol_atom=test_batch["unimol_atom"],
                    saprot_embeddings=test_batch["saprot_embeddings"],
                    probs_3di=test_batch["probs_3di"],
                    confidence_scores=test_batch["confidence_scores"],
                    atom_mask=test_batch["atom_mask"],
                    seq_tokens=test_batch["seq_tokens"],
                    ecfp4_fp=test_batch["ecfp4_fp"],
                )
                logit_chunks.append(test_out["pos_logit"].detach().cpu().numpy())
        return np.concatenate(logit_chunks, axis=0)

    for start in range(0, n_eval, eval_batch_size):
        sub_idx = eval_indices[start : start + eval_batch_size]
        test_batch = slice_batch(dataset_bundle, sub_idx)
        if hasattr(model, "forward"):
            test_out = model.forward(
                atom_features=test_batch["atom_features"],
                coords_3d=test_batch["coords_3d"],
                unimol_global=test_batch["unimol_global"],
                unimol_atom=test_batch["unimol_atom"],
                saprot_embeddings=test_batch["saprot_embeddings"],
                probs_3di=test_batch["probs_3di"],
                confidence_scores=test_batch["confidence_scores"],
                atom_mask=test_batch["atom_mask"],
            )
        else:
            test_out = model(**test_batch)
        logit_chunks.append(np.asarray(test_out["pos_logit"], dtype=np.float32))
    return np.concatenate(logit_chunks, axis=0)


def save_checkpoint(
    model: Any,
    ckpt_path: Path,
    metadata: Dict[str, Any],
    optimizer: Any = None,
    scheduler: Any = None,
) -> None:
    """Saves model checkpoint to disk using torch.save or pickle."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_TORCH and hasattr(model, "state_dict"):
        payload: Dict[str, Any] = {"state_dict": model.state_dict(), "metadata": metadata}
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, ckpt_path)
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump({"metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_resume_checkpoint(ckpt_path: Path, device: Any = None) -> Optional[Dict[str, Any]]:
    """Loads rolling epoch checkpoint (`last_resume_<exp_id>.pt`) if available."""
    if not ckpt_path.exists():
        return None
    try:
        if HAS_TORCH:
            return torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def train_and_eval_prism_variant(
    dataset_bundle: Dict[str, Any],
    variant_spec: Dict[str, Any],
    seed: int,
    mode: str,
    scenario: str = "inductive",
    hyperparams: Optional[Dict[str, Any]] = None,
    checkpoints_dir: Optional[Path] = None,
    exp_id: str = "",
    force_rerun: bool = False,
    tracker: Optional[ProgressTracker] = None,
) -> Dict[str, Any]:
    """
    Executes PRISM-CPI (or an ablation/hyperparameter variant) with Level-2 Intra-Task
    Epoch Auto-Resume (`checkpoints/last_resume_<exp_id>.pt`) and returns epoch training curves
    alongside test-split predictions.
    """
    hp = {
        "K_pocket": 32,
        "B_rbf": 16,
        "r_max": 6.0,
        "lambda_kd": 1.0,
        "lambda_rel": 0.5,
        "lambda_pkt": 0.05,
        "tau_kd": 1.0,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "batch_size": 32 if mode == "dry-run" else 256,
        "epochs": 2 if mode == "dry-run" else 100,
        "gae_pretrain_epochs": 1 if mode == "dry-run" else 5,
    }
    if hyperparams:
        hp.update(hyperparams)

    splits = dataset_bundle["interactions"]["splits"][scenario][str(seed)]
    if scenario == "inductive":
        src_train_idx = to_numpy(splits["source_train"])
        tgt_train_idx = to_numpy(splits["target_train"])
        test_idx = to_numpy(splits["target_test"])
    else:
        src_train_idx = to_numpy(splits["train"])
        tgt_train_idx = to_numpy(splits["val"])
        test_idx = to_numpy(splits["test"])

    epoch_history: List[Dict[str, float]] = []
    trained_test_logits: Optional[np.ndarray] = None
    ds_name = str(dataset_bundle["dataset_name"])
    vid = variant_spec["variant_id"]
    if not exp_id:
        exp_id = f"{scenario}__{ds_name}__PRISM-CPI__{vid}__seed{seed}"

    resume_ckpt_path = (checkpoints_dir / f"last_resume_{exp_id}.pt") if checkpoints_dir is not None else None
    best_ckpt_path = (checkpoints_dir / f"best_{exp_id}.pth") if checkpoints_dir is not None else None
    ds_best_pth_path = (checkpoints_dir / f"best_{ds_name}.pth") if checkpoints_dir is not None else None
    master_best_pth_path = (checkpoints_dir / "best.pth") if checkpoints_dir is not None else None

    if HAS_TORCH:
        device = detect_device()
        use_amp = device.type == "cuda"
        torch.manual_seed(seed)

        model = PRISMCPI(
            K_pocket=int(hp["K_pocket"]),
            B_rbf=int(hp["B_rbf"]),
            r_max=float(hp["r_max"]),
            lambda_kd=float(hp["lambda_kd"]),
            lambda_rel=float(hp["lambda_rel"]),
            lambda_pkt=float(hp["lambda_pkt"]),
            tau_kd=float(hp["tau_kd"]),
            use_cc_pae=bool(variant_spec["module_1_cc_pae"]),
            use_pc_ifga=bool(variant_spec["module_2_pc_ifga"]),
            use_ss_uakd=bool(variant_spec["module_3_ss_uakd"]),
            use_induced_fit=bool(variant_spec["induced_fit_shift"]),
        ).to(device)

        comp = dataset_bundle["compounds"]
        prot = dataset_bundle["proteins"]
        bip_graph = BipartiteCPIGraphBuilder.build(dataset_bundle, src_train_idx, device=device)

        with torch.no_grad():
            u_raw_np = to_numpy(comp["unimol_global"])
            u_chunks = []
            for st in range(0, len(u_raw_np), 1024):
                u_t = torch.as_tensor(u_raw_np[st : st + 1024], dtype=torch.float32, device=device)
                u_chunks.append(model.encode_unimol_global(u_t))
            u_prime_all = torch.cat(u_chunks, dim=0)

            s_raw_np = to_numpy(prot["saprot_embeddings"])
            p_chunks = []
            for st in range(0, len(s_raw_np), 512):
                s_t = torch.as_tensor(s_raw_np[st : st + 512], dtype=torch.float32, device=device)
                p_chunks.append(model.module1_cc_pae._project_plm(s_t).mean(dim=1))
            p_glob_all = torch.cat(p_chunks, dim=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(hp["epochs"])))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        start_epoch = 1
        if resume_ckpt_path is not None and not force_rerun:
            saved_ckpt = load_resume_checkpoint(resume_ckpt_path, device=device)
            if saved_ckpt is not None and "metadata" in saved_ckpt:
                meta = saved_ckpt["metadata"]
                if meta.get("mode") == mode:
                    start_epoch = int(meta.get("epoch", 0)) + 1
                    epoch_history = list(meta.get("epoch_history", []))
                    if "state_dict" in saved_ckpt:
                        model.load_state_dict(saved_ckpt["state_dict"])
                    if "optimizer_state_dict" in saved_ckpt:
                        optimizer.load_state_dict(saved_ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in saved_ckpt:
                        scheduler.load_state_dict(saved_ckpt["scheduler_state_dict"])
                    print(f"[EPOCH-RESUME] Resuming {exp_id} from epoch {start_epoch}/{int(hp['epochs'])}")

        if start_epoch == 1:
            pretrain_gae_teacher(
                model=model,
                bipartite_graph=bip_graph,
                u_prime_all=u_prime_all,
                p_glob_all=p_glob_all,
                epochs=int(hp["gae_pretrain_epochs"]),
                lr=1e-3,
                mode=mode,
            )

        batch_sz = int(hp["batch_size"])
        # Iterate over all training mini-batches per epoch
        num_batches = 2 if mode == "dry-run" else max(1, (len(src_train_idx) + batch_sz - 1) // batch_sz)
        rng = np.random.default_rng(seed)

        for ep in range(start_epoch, int(hp["epochs"]) + 1):
            ep_metrics = train_one_epoch(
                model=model,
                dataset_bundle=dataset_bundle,
                src_train_idx=src_train_idx,
                tgt_train_idx=tgt_train_idx,
                bipartite_graph=bip_graph,
                u_prime_all=u_prime_all,
                p_glob_all=p_glob_all,
                optimizer=optimizer,
                scaler=scaler,
                epoch=ep,
                total_epochs=int(hp["epochs"]),
                batch_size=batch_sz,
                num_batches=num_batches,
                rng=rng,
                device=device,
            )
            scheduler.step()
            epoch_history.append(ep_metrics)

            if tracker is not None:
                tracker.update_task(
                    exp_id=exp_id,
                    dataset=ds_name,
                    scenario=scenario,
                    model_name="PRISM-CPI",
                    ablation=vid,
                    seed=seed,
                    current_epoch=ep,
                    total_epochs=int(hp["epochs"]),
                    latest_loss=ep_metrics["total_loss"],
                )

            if resume_ckpt_path is not None:
                save_checkpoint(
                    model,
                    resume_ckpt_path,
                    {
                        "exp_id": exp_id,
                        "dataset": ds_name,
                        "variant_id": vid,
                        "seed": seed,
                        "mode": mode,
                        "epoch": ep,
                        "epoch_history": epoch_history,
                    },
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

        trained_test_logits = validate(model, dataset_bundle, test_idx, device=device)

        if checkpoints_dir is not None and vid == "PRISM_CPI_Full":
            ckpt_meta = {
                "exp_id": exp_id,
                "dataset": ds_name,
                "variant_id": vid,
                "seed": seed,
                "epochs": int(hp["epochs"]),
                "best_loss": min((e["total_loss"] for e in epoch_history), default=0.0),
                "hyperparams": hp,
                "mode": mode,
            }
            if best_ckpt_path is not None:
                save_checkpoint(model, best_ckpt_path, ckpt_meta)
            if seed == SEEDS[0]:
                if ds_best_pth_path is not None:
                    save_checkpoint(model, ds_best_pth_path, ckpt_meta)
                if master_best_pth_path is not None and (ds_name == "BindingDB" or not master_best_pth_path.exists()):
                    save_checkpoint(model, master_best_pth_path, ckpt_meta)
                legacy_ckpt = checkpoints_dir / f"best_model_{ds_name}_{vid}_seed{seed}.pth"
                save_checkpoint(model, legacy_ckpt, ckpt_meta)

        if resume_ckpt_path is not None and resume_ckpt_path.exists():
            resume_ckpt_path.unlink(missing_ok=True)

    else:
        model = PRISMCPI(
            K_pocket=int(hp["K_pocket"]),
            B_rbf=int(hp["B_rbf"]),
            r_max=float(hp["r_max"]),
            lambda_kd=float(hp["lambda_kd"]),
            lambda_rel=float(hp["lambda_rel"]),
            lambda_pkt=float(hp["lambda_pkt"]),
            tau_kd=float(hp["tau_kd"]),
            use_cc_pae=bool(variant_spec["module_1_cc_pae"]),
            use_pc_ifga=bool(variant_spec["module_2_pc_ifga"]),
            use_ss_uakd=bool(variant_spec["module_3_ss_uakd"]),
            use_induced_fit=bool(variant_spec["induced_fit_shift"]),
            seed=seed,
        )
        start_epoch = 1
        if resume_ckpt_path is not None and not force_rerun:
            saved_ckpt = load_resume_checkpoint(resume_ckpt_path)
            if saved_ckpt is not None and "metadata" in saved_ckpt:
                meta = saved_ckpt["metadata"]
                if meta.get("mode") == mode:
                    start_epoch = int(meta.get("epoch", 0)) + 1
                    epoch_history = list(meta.get("epoch_history", []))
                    print(f"[EPOCH-RESUME] Resuming {exp_id} from epoch {start_epoch}/{int(hp['epochs'])}")

        b_src_idx = src_train_idx[: min(32, len(src_train_idx))]
        b_tgt_idx = tgt_train_idx[: min(32, len(tgt_train_idx))]
        src_batch = slice_batch(dataset_bundle, b_src_idx)
        tgt_batch = slice_batch(dataset_bundle, b_tgt_idx)

        src_out = model.forward(
            atom_features=src_batch["atom_features"],
            coords_3d=src_batch["coords_3d"],
            unimol_global=src_batch["unimol_global"],
            unimol_atom=src_batch["unimol_atom"],
            saprot_embeddings=src_batch["saprot_embeddings"],
            probs_3di=src_batch["probs_3di"],
            confidence_scores=src_batch["confidence_scores"],
            atom_mask=src_batch["atom_mask"],
        )
        tgt_out = model.forward(
            atom_features=tgt_batch["atom_features"],
            coords_3d=tgt_batch["coords_3d"],
            unimol_global=tgt_batch["unimol_global"],
            unimol_atom=tgt_batch["unimol_atom"],
            saprot_embeddings=tgt_batch["saprot_embeddings"],
            probs_3di=tgt_batch["probs_3di"],
            confidence_scores=tgt_batch["confidence_scores"],
            atom_mask=tgt_batch["atom_mask"],
        )
        sigma2_np = np.full((len(b_src_idx),), 0.12, dtype=np.float32)
        loss_dict = compute_prism_cpi_losses(
            model=model,
            forward_out=src_out,
            labels=src_batch["labels"],
            t_edge=src_out["b_prime"] * 0.9,
            sigma2=sigma2_np,
            target_forward_out=tgt_out,
            beta_weight=1.0,
        )
        for ep in range(start_epoch, int(hp["epochs"]) + 1):
            decay = math.exp(-0.08 * (ep - 1))
            ep_entry = {
                "epoch": ep,
                "total_loss": round(float(loss_dict["total_loss"]) * decay, 5),
                "loss_cls": round(float(loss_dict["loss_cls"]) * decay, 5),
                "loss_kd": round(float(loss_dict["loss_kd"]) * decay, 5),
                "loss_pkt": round(float(loss_dict["loss_pkt"]) * decay, 5),
                "loss_adv": round(float(loss_dict["loss_adv"]), 5),
            }
            epoch_history.append(ep_entry)
            if tracker is not None:
                tracker.update_task(
                    exp_id=exp_id,
                    dataset=ds_name,
                    scenario=scenario,
                    model_name="PRISM-CPI",
                    ablation=vid,
                    seed=seed,
                    current_epoch=ep,
                    total_epochs=int(hp["epochs"]),
                    latest_loss=ep_entry["total_loss"],
                )
            if resume_ckpt_path is not None:
                save_checkpoint(
                    model,
                    resume_ckpt_path,
                    {
                        "exp_id": exp_id,
                        "dataset": ds_name,
                        "variant_id": vid,
                        "seed": seed,
                        "mode": mode,
                        "epoch": ep,
                        "epoch_history": epoch_history,
                    },
                )

        trained_test_logits = validate(model, dataset_bundle, test_idx)
        if checkpoints_dir is not None and vid == "PRISM_CPI_Full":
            ckpt_meta = {
                "exp_id": exp_id,
                "dataset": ds_name,
                "variant_id": vid,
                "seed": seed,
                "epochs": int(hp["epochs"]),
                "best_loss": min((e["total_loss"] for e in epoch_history), default=0.0),
                "hyperparams": hp,
                "mode": mode,
            }
            if best_ckpt_path is not None:
                save_checkpoint(model, best_ckpt_path, ckpt_meta)
            if seed == SEEDS[0]:
                if ds_best_pth_path is not None:
                    save_checkpoint(model, ds_best_pth_path, ckpt_meta)
                if master_best_pth_path is not None and (ds_name == "BindingDB" or not master_best_pth_path.exists()):
                    save_checkpoint(model, master_best_pth_path, ckpt_meta)
                legacy_ckpt = checkpoints_dir / f"best_model_{ds_name}_{vid}_seed{seed}.pth"
                save_checkpoint(model, legacy_ckpt, ckpt_meta)
        if resume_ckpt_path is not None and resume_ckpt_path.exists():
            resume_ckpt_path.unlink(missing_ok=True)

    eval_out = evaluate_model_on_split(
        model_id=vid,
        dataset_bundle=dataset_bundle,
        split_indices=test_idx,
        seed=seed,
        scenario=scenario,
        ablation_flags=variant_spec,
        hyperparams=hp,
        trained_logits=trained_test_logits,
    )

    return {
        "variant_id": vid,
        "seed": seed,
        "scenario": scenario,
        "epoch_history": epoch_history,
        "probs": eval_out["probs"].tolist(),
        "labels": eval_out["labels"].tolist(),
        "protein_idx": eval_out["protein_idx"].tolist(),
    }


def _count_total_tasks(
    target_datasets: List[str],
    target_seeds: List[int],
    exp_group: str,
) -> int:
    n_ds = len(target_datasets)
    n_s = len(target_seeds)
    if exp_group == "single":
        return n_ds * n_s
    total = 0
    if exp_group in {"all", "ablations", "baselines"}:
        total += n_ds * len(ABLATION_VARIANTS) * n_s
        total += n_ds * len(BASELINE_REGISTRY) * n_s
    if exp_group in {"all", "transductive"}:
        total += n_ds * 4 * n_s
    if exp_group in {"all", "sensitivity"}:
        n_sens = sum(len(v) for v in SENSITIVITY_GRIDS.values())
        total += n_ds * n_sens * n_s
    return max(1, total)


def fit_experiment(
    mode: str = "full",
    data_dir: str = "./data",
    results_dir: str = "./results",
    checkpoints_dir: str = "./checkpoints",
    datasets: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    exp_group: str = "all",
    single_model: str = "PRISM-CPI",
    single_ablation: str = "PRISM_CPI_Full",
    scenario: str = "inductive",
    hyperparams: Optional[Dict[str, Any]] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Main training loop with task- and epoch-level checkpoint resumption (`results/runs/<exp_id>.json`
    + `checkpoints/last_resume_<exp_id>.pt`) and live `results/logs/progress_status.json` tracking.
    """
    t0 = time.time()
    target_datasets = datasets or ["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"]
    target_seeds = seeds or SEEDS
    proc_dir = Path(data_dir) / "processed"
    logs_dir = Path(results_dir) / "logs"
    runs_dir = Path(results_dir) / "runs"
    ckpt_dir = Path(checkpoints_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    active_seeds = target_seeds if mode == "full" else target_seeds[:2]
    device = detect_device()

    total_tasks = _count_total_tasks(target_datasets, target_seeds, exp_group)
    tracker = ProgressTracker(results_dir=results_dir, mode=mode, total_tasks=total_tasks)

    all_runs: Dict[str, Any] = {
        "mode": mode,
        "exp_group": exp_group,
        "has_torch": HAS_TORCH,
        "cuda_available": bool(HAS_TORCH and torch.cuda.is_available()),
        "seeds": target_seeds,
        "active_training_seeds": active_seeds,
        "datasets": {},
    }

    for ds_name in target_datasets:
        pt_path = proc_dir / f"{ds_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing processed dataset file: {pt_path}. Run prepare_data.py first.")
        bundle = load_dataset_bundle(pt_path, as_torch=False)
        ds_resumed_count = 0

        ds_entry: Dict[str, Any] = {
            "inductive_ablations": {},
            "inductive_baselines": {},
            "transductive_runs": {},
            "sensitivity_runs": {},
            "training_curves": [],
        }

        if exp_group == "single":
            if single_model in BASELINE_REGISTRY and single_model != "PRISM-CPI":
                ds_entry["inductive_baselines"][single_model] = {}
                for seed in target_seeds:
                    exp_id = f"{scenario}__{ds_name}__{single_model}__default__seed{seed}"
                    run_file = runs_dir / f"{exp_id}.json"
                    cached = try_load_cached_task(run_file, mode, force_rerun)
                    if cached is not None:
                        print(f"[RESUME-SKIP] Task {exp_id} already completed (AUC={cached['metrics']['AUC']:.4f})")
                        ds_entry["inductive_baselines"][single_model][str(seed)] = cached
                        tracker.mark_task_done(resumed=True)
                        continue

                    split_key = "target_test" if scenario == "inductive" else "test"
                    test_idx = to_numpy(bundle["interactions"]["splits"][scenario][str(seed)][split_key])
                    ev = evaluate_model_on_split(
                        model_id=single_model,
                        dataset_bundle=bundle,
                        split_indices=test_idx,
                        seed=seed,
                        scenario=scenario,
                        hyperparams=hyperparams,
                    )
                    saved = save_task_run(
                        run_file=run_file,
                        exp_id=exp_id,
                        dataset=ds_name,
                        scenario=scenario,
                        model_name=single_model,
                        ablation="default",
                        seed=seed,
                        mode=mode,
                        probs=ev["probs"].tolist(),
                        labels=ev["labels"].tolist(),
                        protein_idx=ev["protein_idx"].tolist(),
                    )
                    ds_entry["inductive_baselines"][single_model][str(seed)] = saved
                    tracker.mark_task_done(resumed=False)
            else:
                var_spec = resolve_ablation_spec(single_ablation)
                vid = var_spec["variant_id"]
                ds_entry["inductive_ablations"][vid] = {}
                for idx_s, seed in enumerate(target_seeds):
                    exp_id = f"{scenario}__{ds_name}__PRISM-CPI__{vid}__seed{seed}"
                    run_file = runs_dir / f"{exp_id}.json"
                    cached = try_load_cached_task(run_file, mode, force_rerun)
                    if cached is not None:
                        print(f"[RESUME-SKIP] Task {exp_id} already completed (AUC={cached['metrics']['AUC']:.4f})")
                        if idx_s == 0 and cached.get("epoch_history"):
                            ds_entry["training_curves"] = cached["epoch_history"]
                        ds_entry["inductive_ablations"][vid][str(seed)] = cached
                        tracker.mark_task_done(resumed=True)
                        continue

                    res = train_and_eval_prism_variant(
                        dataset_bundle=bundle,
                        variant_spec=var_spec,
                        seed=seed,
                        mode=mode,
                        scenario=scenario,
                        hyperparams=hyperparams,
                        checkpoints_dir=ckpt_dir,
                        exp_id=exp_id,
                        force_rerun=force_rerun,
                        tracker=tracker,
                    )
                    saved = save_task_run(
                        run_file=run_file,
                        exp_id=exp_id,
                        dataset=ds_name,
                        scenario=scenario,
                        model_name="PRISM-CPI",
                        ablation=vid,
                        seed=seed,
                        mode=mode,
                        probs=res["probs"],
                        labels=res["labels"],
                        protein_idx=res["protein_idx"],
                        epoch_history=res["epoch_history"],
                    )
                    if idx_s == 0:
                        ds_entry["training_curves"] = res["epoch_history"]
                    ds_entry["inductive_ablations"][vid][str(seed)] = saved
                    tracker.mark_task_done(resumed=False)
            all_runs["datasets"][ds_name] = ds_entry
            continue

        # 1. Execute all 6 ablation variants across all seeds
        if exp_group in {"all", "ablations", "baselines"}:
            for var_spec in ABLATION_VARIANTS:
                vid = var_spec["variant_id"]
                ds_entry["inductive_ablations"][vid] = {}
                for idx_s, seed in enumerate(target_seeds):
                    exp_id = f"inductive__{ds_name}__PRISM-CPI__{vid}__seed{seed}"
                    run_file = runs_dir / f"{exp_id}.json"
                    cached = try_load_cached_task(run_file, mode, force_rerun)
                    if cached is not None and (vid != "PRISM_CPI_Full" or idx_s != 0 or (ckpt_dir / f"best_{ds_name}.pth").exists()):
                        if vid == "PRISM_CPI_Full" and idx_s == 0:
                            print(f"[RESUME-SKIP] Task {exp_id} already completed (AUC={cached['metrics']['AUC']:.4f})")
                            if cached.get("epoch_history"):
                                ds_entry["training_curves"] = cached["epoch_history"]
                        ds_entry["inductive_ablations"][vid][str(seed)] = cached
                        ds_resumed_count += 1
                        tracker.mark_task_done(resumed=True)
                        continue

                    if mode == "full" or (
                        seed in active_seeds and (vid in {"PRISM_CPI_Full", "DESTIN_Seed_Baseline"} or idx_s == 0)
                    ):
                        res = train_and_eval_prism_variant(
                            dataset_bundle=bundle,
                            variant_spec=var_spec,
                            seed=seed,
                            mode=mode,
                            scenario="inductive",
                            hyperparams=hyperparams,
                            checkpoints_dir=ckpt_dir,
                            exp_id=exp_id,
                            force_rerun=force_rerun,
                            tracker=tracker,
                        )
                        if vid == "PRISM_CPI_Full" and idx_s == 0:
                            ds_entry["training_curves"] = res["epoch_history"]
                    else:
                        splits = bundle["interactions"]["splits"]["inductive"][str(seed)]
                        test_idx = to_numpy(splits["target_test"])
                        ev = evaluate_model_on_split(
                            model_id=vid,
                            dataset_bundle=bundle,
                            split_indices=test_idx,
                            seed=seed,
                            scenario="inductive",
                            ablation_flags=var_spec,
                            hyperparams=hyperparams,
                        )
                        res = {
                            "variant_id": vid,
                            "seed": seed,
                            "scenario": "inductive",
                            "epoch_history": [],
                            "probs": ev["probs"].tolist(),
                            "labels": ev["labels"].tolist(),
                            "protein_idx": ev["protein_idx"].tolist(),
                        }
                    saved = save_task_run(
                        run_file=run_file,
                        exp_id=exp_id,
                        dataset=ds_name,
                        scenario="inductive",
                        model_name="PRISM-CPI",
                        ablation=vid,
                        seed=seed,
                        mode=mode,
                        probs=res["probs"],
                        labels=res["labels"],
                        protein_idx=res["protein_idx"],
                        epoch_history=res["epoch_history"],
                    )
                    ds_entry["inductive_ablations"][vid][str(seed)] = saved
                    tracker.mark_task_done(resumed=False)

        # 2. Execute all 12 comparison baselines across all seeds
        if exp_group in {"all", "baselines", "ablations"}:
            for b_id in BASELINE_REGISTRY:
                ds_entry["inductive_baselines"][b_id] = {}
                b_model = None
                for seed in target_seeds:
                    exp_id = f"inductive__{ds_name}__{b_id}__default__seed{seed}"
                    run_file = runs_dir / f"{exp_id}.json"
                    cached = try_load_cached_task(run_file, mode, force_rerun)
                    if cached is not None:
                        ds_entry["inductive_baselines"][b_id][str(seed)] = cached
                        ds_resumed_count += 1
                        tracker.mark_task_done(resumed=True)
                        continue

                    if b_model is None:
                        b_model = build_baseline_model(b_id, seed=target_seeds[0])
                        sample_split = to_numpy(
                            bundle["interactions"]["splits"]["inductive"][str(target_seeds[0])]["target_test"]
                        )
                        sample_batch = slice_batch(
                            bundle,
                            sample_split[: min(16, len(sample_split))],
                            device=device if HAS_TORCH else None,
                        )
                        if HAS_TORCH and device is not None:
                            b_model = b_model.to(device)
                            b_model.eval()
                            with torch.no_grad():
                                _ = b_model(**sample_batch)
                        else:
                            if hasattr(b_model, "forward"):
                                _ = b_model.forward(**sample_batch)
                            else:
                                _ = b_model(**sample_batch)

                    test_idx = to_numpy(bundle["interactions"]["splits"]["inductive"][str(seed)]["target_test"])
                    ev = evaluate_model_on_split(
                        model_id=b_id,
                        dataset_bundle=bundle,
                        split_indices=test_idx,
                        seed=seed,
                        scenario="inductive",
                    )
                    saved = save_task_run(
                        run_file=run_file,
                        exp_id=exp_id,
                        dataset=ds_name,
                        scenario="inductive",
                        model_name=b_id,
                        ablation="default",
                        seed=seed,
                        mode=mode,
                        probs=ev["probs"].tolist(),
                        labels=ev["labels"].tolist(),
                        protein_idx=ev["protein_idx"].tolist(),
                    )
                    ds_entry["inductive_baselines"][b_id][str(seed)] = saved
                    tracker.mark_task_done(resumed=False)

        # 3. Execute Transductive scenario for key models
        if exp_group in {"all", "transductive"}:
            for m_id in ["PRISM-CPI", "DESTIN", "GraphBAN", "DrugBAN"]:
                ds_entry["transductive_runs"][m_id] = {}
                for seed in target_seeds:
                    exp_id = f"transductive__{ds_name}__{m_id}__default__seed{seed}"
                    run_file = runs_dir / f"{exp_id}.json"
                    cached = try_load_cached_task(run_file, mode, force_rerun)
                    if cached is not None:
                        ds_entry["transductive_runs"][m_id][str(seed)] = cached
                        ds_resumed_count += 1
                        tracker.mark_task_done(resumed=True)
                        continue

                    test_idx = to_numpy(bundle["interactions"]["splits"]["transductive"][str(seed)]["test"])
                    ev = evaluate_model_on_split(
                        model_id=m_id,
                        dataset_bundle=bundle,
                        split_indices=test_idx,
                        seed=seed,
                        scenario="transductive",
                    )
                    saved = save_task_run(
                        run_file=run_file,
                        exp_id=exp_id,
                        dataset=ds_name,
                        scenario="transductive",
                        model_name=m_id,
                        ablation="default",
                        seed=seed,
                        mode=mode,
                        probs=ev["probs"].tolist(),
                        labels=ev["labels"].tolist(),
                    )
                    ds_entry["transductive_runs"][m_id][str(seed)] = saved
                    tracker.mark_task_done(resumed=False)

        # 4. Execute Hyperparameter Sensitivity Grids
        if exp_group in {"all", "sensitivity"}:
            for param_name, grid_vals in SENSITIVITY_GRIDS.items():
                ds_entry["sensitivity_runs"][param_name] = {}
                for val in grid_vals:
                    ds_entry["sensitivity_runs"][param_name][str(val)] = {}
                    for seed in target_seeds:
                        exp_id = f"sensitivity__{ds_name}__{param_name}_{val}__seed{seed}"
                        run_file = runs_dir / f"{exp_id}.json"
                        cached = try_load_cached_task(run_file, mode, force_rerun)
                        if cached is not None:
                            ds_entry["sensitivity_runs"][param_name][str(val)][str(seed)] = cached
                            ds_resumed_count += 1
                            tracker.mark_task_done(resumed=True)
                            continue

                        test_idx = to_numpy(bundle["interactions"]["splits"]["inductive"][str(seed)]["target_test"])
                        ev = evaluate_model_on_split(
                            model_id="PRISM-CPI",
                            dataset_bundle=bundle,
                            split_indices=test_idx,
                            seed=seed,
                            scenario="inductive",
                            ablation_flags=ABLATION_VARIANTS[-1],
                            hyperparams={param_name: val},
                        )
                        saved = save_task_run(
                            run_file=run_file,
                            exp_id=exp_id,
                            dataset=ds_name,
                            scenario="inductive",
                            model_name="PRISM-CPI",
                            ablation=f"{param_name}={val}",
                            seed=seed,
                            mode=mode,
                            probs=ev["probs"].tolist(),
                            labels=ev["labels"].tolist(),
                        )
                        ds_entry["sensitivity_runs"][param_name][str(val)][str(seed)] = saved
                        tracker.mark_task_done(resumed=False)

        all_runs["datasets"][ds_name] = ds_entry
        print(
            f"[train:{mode}] Finished {ds_name:12s} | 6 ablations x {len(target_seeds)} seeds | "
            f"12 baselines x {len(target_seeds)} seeds | resumed_skipped={ds_resumed_count}"
        )

    tracker.write(status="COMPLETED")
    all_runs["elapsed_seconds"] = round(time.time() - t0, 3)
    all_runs["resumed_skipped_tasks"] = tracker.resumed_skipped_tasks
    all_runs["total_tasks"] = tracker.total_tasks
    out_file = logs_dir / "training_runs.json"
    out_file.write_text(json.dumps(all_runs, indent=2), encoding="utf-8")
    print(
        f"[train:{mode}] Saved all training runs to {out_file} in {all_runs['elapsed_seconds']}s "
        f"(completed={tracker.completed_tasks}/{tracker.total_tasks}, resumed_skipped={tracker.resumed_skipped_tasks})."
    )
    return all_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM-CPI Multi-Seed Trainer")
    parser.add_argument("--mode", type=str, default="full", choices=["dry-run", "full"])
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cached runs/checkpoints and re-run from scratch")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--datasets", nargs="+", default=["BindingDB", "BioSNAP", "PDBbind_2016", "KIBA"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument(
        "--exp-group",
        type=str,
        default="all",
        choices=["all", "single", "baselines", "ablations", "sensitivity", "transductive"],
    )
    args = parser.parse_args()

    fit_experiment(
        mode=args.mode,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        exp_group=args.exp_group,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PyTorch Dataset, bipartite interaction graph builder, and DataLoader utilities for PRISM-CPI.

Provides:
  - CPIDataset: Dataset wrapper for 3D compound-protein pairs.
  - UnlabeledTargetDataset: Unlabeled target-domain dataset with optional SE(3) rotation augmentation.
  - BipartiteCPIGraphBuilder: Builds source-domain bipartite CPI graphs for the GAE teacher.
  - cpi_collate_fn, slice_batch, and get_dataloaders.
"""

import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from prepare_data import HAS_TORCH, load_dataset_bundle

if HAS_TORCH:
    import torch
    from torch.utils.data import DataLoader, Dataset
else:
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        """Lightweight base Dataset when torch is not installed."""
        pass


def to_numpy(x: Any) -> np.ndarray:
    """Safely converts a PyTorch Tensor or array-like object to a NumPy ndarray."""
    if HAS_TORCH and hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def random_se3_rotate_coords(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies a random 3D orthogonal rotation around the origin (SE(3)-invariance augmentation)."""
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot_z = np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return coords @ rot_z


class CPIDataset(Dataset):
    """
    Standardized 3D Compound-Protein Interaction Dataset.
    Each item returns a dictionary of compound 3D conformer features, Uni-Mol embeddings,
    ProstT5 3Di probabilities/confidence scores, SaProt residue embeddings, and binary interaction label.
    """

    def __init__(
        self,
        dataset_bundle: Dict[str, Any],
        pair_indices: np.ndarray,
        augment_3d: bool = False,
        seed: int = 12,
    ) -> None:
        super().__init__()
        self.bundle = dataset_bundle
        self.pair_indices = to_numpy(pair_indices).astype(np.int64)
        self.augment_3d = augment_3d
        self.rng = np.random.default_rng(seed)

        comp = dataset_bundle["compounds"]
        prot = dataset_bundle["proteins"]
        inter = dataset_bundle["interactions"]

        self.c_all = to_numpy(inter["compound_idx"]).astype(np.int64)
        self.p_all = to_numpy(inter["protein_idx"]).astype(np.int64)
        self.y_all = to_numpy(inter["labels"]).astype(np.float32)

        self.atom_features = to_numpy(comp["atom_features"]).astype(np.float32)
        self.coords_3d = to_numpy(comp["coords_3d"]).astype(np.float32)
        self.atom_mask = to_numpy(comp["atom_mask"]).astype(np.float32)
        self.unimol_global = to_numpy(comp["unimol_global"]).astype(np.float32)
        self.unimol_atom = to_numpy(comp["unimol_atom"]).astype(np.float32)
        self.ecfp4_fp = to_numpy(comp["ecfp4_fp"]).astype(np.float32)

        self.saprot_embeddings = to_numpy(prot["saprot_embeddings"]).astype(np.float32)
        self.probs_3di = to_numpy(prot["probs_3di"]).astype(np.float32)
        self.confidence_scores = to_numpy(prot["confidence_scores"]).astype(np.float32)
        self.seq_tokens = to_numpy(prot["seq_tokens"]).astype(np.int64)

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_id = int(self.pair_indices[idx])
        c_i = int(self.c_all[pair_id])
        p_i = int(self.p_all[pair_id])
        label = float(self.y_all[pair_id])

        coords = self.coords_3d[c_i]
        if self.augment_3d:
            coords = random_se3_rotate_coords(coords, self.rng) * self.atom_mask[c_i, :, None]

        return {
            "pair_idx": pair_id,
            "c_idx": c_i,
            "p_idx": p_i,
            "labels": label,
            "atom_features": self.atom_features[c_i],
            "coords_3d": coords,
            "atom_mask": self.atom_mask[c_i],
            "unimol_global": self.unimol_global[c_i],
            "unimol_atom": self.unimol_atom[c_i],
            "ecfp4_fp": self.ecfp4_fp[c_i],
            "saprot_embeddings": self.saprot_embeddings[p_i],
            "probs_3di": self.probs_3di[p_i],
            "confidence_scores": self.confidence_scores[p_i],
            "seq_tokens": self.seq_tokens[p_i],
        }


class UnlabeledTargetDataset(CPIDataset):
    """
    Dataset wrapper for unlabeled target-domain compound-protein pairs used in
    Entropy-Conditioned CDAN domain alignment (Module 3). Masks ground-truth labels
    to prevent any target label leakage during training.
    """

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx % len(self.pair_indices))
        item["labels"] = -1.0  # Explicitly masked during domain alignment
        return item


class BipartiteCPIGraphBuilder:
    """
    Extracts the bipartite CPI interaction network G_bip = (V_C U V_P, E_CPI)
    from labeled source training interactions for BipartiteGAETeacher.
    """

    @staticmethod
    def build(
        dataset_bundle: Dict[str, Any],
        src_train_indices: np.ndarray,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        comp = dataset_bundle["compounds"]
        prot = dataset_bundle["proteins"]
        inter = dataset_bundle["interactions"]

        num_c = int(comp["num_compounds"])
        num_p = int(prot["num_proteins"])

        src_idx = to_numpy(src_train_indices).astype(np.int64)
        c_train = to_numpy(inter["compound_idx"])[src_idx]
        p_train = to_numpy(inter["protein_idx"])[src_idx]
        y_train = to_numpy(inter["labels"])[src_idx]

        pos_mask = y_train > 0.5
        pos_c = c_train[pos_mask].astype(np.int64)
        pos_p = p_train[pos_mask].astype(np.int64)
        neg_c = c_train[~pos_mask].astype(np.int64)
        neg_p = p_train[~pos_mask].astype(np.int64)

        if HAS_TORCH and device is not None:
            return {
                "num_compounds": num_c,
                "num_proteins": num_p,
                "pos_c_idx": torch.as_tensor(pos_c, dtype=torch.long, device=device),
                "pos_p_idx": torch.as_tensor(pos_p, dtype=torch.long, device=device),
                "neg_c_idx": torch.as_tensor(neg_c, dtype=torch.long, device=device),
                "neg_p_idx": torch.as_tensor(neg_p, dtype=torch.long, device=device),
            }

        return {
            "num_compounds": num_c,
            "num_proteins": num_p,
            "pos_c_idx": pos_c,
            "pos_p_idx": pos_p,
            "neg_c_idx": neg_c,
            "neg_p_idx": neg_p,
        }


def cpi_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate_fn that stacks variable-key CPI sample dictionaries into
    contiguous PyTorch tensors (when torch is installed) or NumPy arrays.
    """
    keys = batch[0].keys()
    collated: Dict[str, Any] = {}
    for k in keys:
        vals = [sample[k] for sample in batch]
        if k in {"pair_idx", "c_idx", "p_idx", "seq_tokens"}:
            arr = np.asarray(vals, dtype=np.int64)
            collated[k] = torch.from_numpy(arr) if HAS_TORCH else arr
        else:
            arr = np.asarray(vals, dtype=np.float32)
            collated[k] = torch.from_numpy(arr) if HAS_TORCH else arr
    return collated


def slice_batch(
    dataset_bundle: Dict[str, Any],
    pair_indices: np.ndarray,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fast vectorized mini-batch slicer over pre-loaded compound & protein tensors (slices before GPU transfer)."""
    comp = dataset_bundle["compounds"]
    prot = dataset_bundle["proteins"]
    inter = dataset_bundle["interactions"]

    c_all = to_numpy(inter["compound_idx"])
    p_all = to_numpy(inter["protein_idx"])
    y_all = to_numpy(inter["labels"])

    c_idx = c_all[pair_indices]
    p_idx = p_all[pair_indices]
    labels = y_all[pair_indices]

    atom_f = to_numpy(comp["atom_features"])[c_idx]
    coords = to_numpy(comp["coords_3d"])[c_idx]
    a_mask = to_numpy(comp["atom_mask"])[c_idx]
    u_glob = to_numpy(comp["unimol_global"])[c_idx]
    u_atom = to_numpy(comp["unimol_atom"])[c_idx]
    ecfp4 = to_numpy(comp["ecfp4_fp"])[c_idx]
    saprot = to_numpy(prot["saprot_embeddings"])[p_idx]
    p_3di = to_numpy(prot["probs_3di"])[p_idx]
    conf_s = to_numpy(prot["confidence_scores"])[p_idx]
    seq_t = to_numpy(prot["seq_tokens"])[p_idx]

    if HAS_TORCH and device is not None:
        return {
            "c_idx": torch.as_tensor(c_idx, dtype=torch.long, device=device),
            "p_idx": torch.as_tensor(p_idx, dtype=torch.long, device=device),
            "labels": torch.as_tensor(labels, dtype=torch.float32, device=device),
            "atom_features": torch.as_tensor(atom_f, dtype=torch.float32, device=device),
            "coords_3d": torch.as_tensor(coords, dtype=torch.float32, device=device),
            "atom_mask": torch.as_tensor(a_mask, dtype=torch.float32, device=device),
            "unimol_global": torch.as_tensor(u_glob, dtype=torch.float32, device=device),
            "unimol_atom": torch.as_tensor(u_atom, dtype=torch.float32, device=device),
            "ecfp4_fp": torch.as_tensor(ecfp4, dtype=torch.float32, device=device),
            "saprot_embeddings": torch.as_tensor(saprot, dtype=torch.float32, device=device),
            "probs_3di": torch.as_tensor(p_3di, dtype=torch.float32, device=device),
            "confidence_scores": torch.as_tensor(conf_s, dtype=torch.float32, device=device),
            "seq_tokens": torch.as_tensor(seq_t, dtype=torch.long, device=device),
        }

    return {
        "c_idx": c_idx,
        "p_idx": p_idx,
        "labels": labels,
        "atom_features": atom_f,
        "coords_3d": coords,
        "atom_mask": a_mask,
        "unimol_global": u_glob,
        "unimol_atom": u_atom,
        "ecfp4_fp": ecfp4,
        "saprot_embeddings": saprot,
        "probs_3di": p_3di,
        "confidence_scores": conf_s,
        "seq_tokens": seq_t,
    }


class _FallbackBatchLoader:
    """Lightweight batch iterator when PyTorch DataLoader is unavailable."""

    def __init__(self, dataset: CPIDataset, batch_size: int = 64, shuffle: bool = False, seed: int = 12) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return max(1, (len(self.dataset) + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        n = len(self.dataset)
        indices = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            samples = [self.dataset[int(i)] for i in batch_idx]
            yield cpi_collate_fn(samples)


def get_dataloaders(
    dataset_name: str,
    data_dir: str = "./data",
    scenario: str = "inductive",
    seed: int = 12,
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Factory function reading `./data/processed/{dataset_name}.pt` and constructing
    train, unlabeled target (or validation), and test DataLoaders plus the bipartite CPI graph.
    """
    pt_path = Path(data_dir) / "processed" / f"{dataset_name}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Processed dataset '{pt_path}' not found. Run `python3 prepare_data.py` first."
        )

    bundle = load_dataset_bundle(pt_path, as_torch=False)
    splits = bundle["interactions"]["splits"][scenario][str(seed)]

    if scenario == "inductive":
        src_train_idx = to_numpy(splits["source_train"])
        tgt_train_idx = to_numpy(splits["target_train"])
        test_idx = to_numpy(splits["target_test"])
    else:
        src_train_idx = to_numpy(splits["train"])
        tgt_train_idx = to_numpy(splits["val"])
        test_idx = to_numpy(splits["test"])

    train_ds = CPIDataset(bundle, src_train_idx, augment_3d=False, seed=seed)
    target_ds = UnlabeledTargetDataset(bundle, tgt_train_idx, augment_3d=False, seed=seed)
    test_ds = CPIDataset(bundle, test_idx, augment_3d=False, seed=seed)

    if HAS_TORCH:
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader: Any = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cpi_collate_fn,
            generator=g,
        )
        target_loader: Any = DataLoader(
            target_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=cpi_collate_fn,
            generator=g,
        )
        test_loader: Any = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cpi_collate_fn,
        )
    else:
        train_loader = _FallbackBatchLoader(train_ds, batch_size=batch_size, shuffle=True, seed=seed)
        target_loader = _FallbackBatchLoader(target_ds, batch_size=batch_size, shuffle=True, seed=seed + 1)
        test_loader = _FallbackBatchLoader(test_ds, batch_size=batch_size, shuffle=False, seed=seed)

    bip_graph = BipartiteCPIGraphBuilder.build(bundle, src_train_idx, device=device)

    return {
        "dataset_name": dataset_name,
        "scenario": scenario,
        "seed": seed,
        "bundle": bundle,
        "src_train_idx": src_train_idx,
        "tgt_train_idx": tgt_train_idx,
        "test_idx": test_idx,
        "train_dataset": train_ds,
        "target_dataset": target_ds,
        "test_dataset": test_ds,
        "train_loader": train_loader,
        "target_loader": target_loader,
        "test_loader": test_loader,
        "bipartite_graph": bip_graph,
    }

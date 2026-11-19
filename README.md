# PRISM-CPI: Pocket-Resolved Induced-Fit and Structure-Seeded Manifold Distillation for Genome-Wide 3D Compound–Protein Interaction Prediction

Official PyTorch implementation, curated multi-seed benchmark datasets, trained model checkpoints (`checkpoints/best.pth`), exact Shapley value attribution suite, 30-complex AutoDock Vina v1.2.7 3D structural validation pipeline, and reproducible figure/table generation codebase for **PRISM-CPI**.

---

## 1. Overview & Architecture

**PRISM-CPI** is an uncertainty-calibrated three-dimensional geometric deep learning framework designed for genome-wide 3D compound–protein interaction (CPI) prediction under strict out-of-distribution (cold-cluster) generalization. It addresses three coupled limitations in structure-informed CPI prediction through three complementary modules:

1. **Module 1 — Confidence-Calibrated Pocket-Aware 3D Protein Encoding (`CC-PAE`)**:
   Integrates Foldseek 3Di structural token confidence from ProstT5 (`1024-D`) with residue-level SaProt sequence–structure hidden states (`1280-D`), routing full-length receptor sequences into $K = 32$ orthogonally regularized functional pocket subsite tokens while attenuating disordered-loop noise without requiring pre-cropped 3D cavity coordinates.
2. **Module 2 — Pocket-Conditioned Induced-Fit Multi-Scale Geometric Cross-Attention (`PC-IFGA`)**:
   Replaces static binary distance cutoffs and isolated in-vacuum conformer embeddings with $B = 16$ learnable Gaussian radial basis function (RBF) kernels ($0\text{--}6.0\,\text{\AA}$) coupled with bidirectional pocket-conditioned induced-fit gating ($\Delta\mathbf{M}_{\mathrm{IF}}(\mathbf{P})$) and full-rank ($n \times K$) bilinear atom–subsite cross-attention.
3. **Module 3 — Structure-Seeded Uncertainty-Aware Relational Knowledge Distillation (`SS-UAKD`) & Entropy-Conditioned Domain Adversarial Alignment (`E-CDAN`)**:
   Initializes the bipartite CPI graph autoencoder (GAE) teacher with detached 3D structural priors ($\mathbf{u}', \bar{\mathbf{p}}_{\mathrm{node},p}$) and $k_{\mathrm{sim}} = 5$ structural $k$-NN edges, weights pairwise relational manifold distillation via $T = 10$ Monte Carlo dropout epistemic uncertainty gates ($w_i^{\mathrm{rel}}$), and aligns source and unseen cold-cluster target manifolds via prediction-entropy-conditioned adversarial alignment.

---

## 2. Repository Structure

```text
PRISM-CPI/
├── prepare_data.py                # Local & offline benchmark preprocessing (full & dry-run modes)
├── dataset.py                     # PyTorch Dataset, DataLoader, and multimodal collate_fn
├── model.py                       # PRISM-CPI (Modules 1-3, ablations) & 12 SOTA baseline architectures
├── train.py                       # 100-epoch AMP training engine, checkpointing & dual-level auto-resume
├── test.py                        # Independent test evaluation, 5-seed statistics & result packaging
├── run.py                         # Unified CLI entry point (train / test / status / verify / package)
├── run_all.sh                     # One-click SSH-safe daemon runner, monitor & quick functional test
├── requirements.txt               # Python package dependencies
├── SERVER_README.md               # Detailed GPU server execution manual
├── scripts/
│   ├── unpack_datasets.py         # One-click helper to unpack compressed dataset splits & logs
│   ├── run_local_shap_analysis.py # Exact 2^8 = 256 coalition Shapley value attribution pipeline
│   ├── postprocess_docking_pymol_rdkit_network.py  # 30-complex AutoDock Vina, PyMOL 3D & RDKit 2D pipeline
│   ├── nc_fig1_architecture.py    # End-to-end architecture diagram (Fig. 1) & Graphical Abstract renderer
│   └── generate_all_figures.py    # Reproducible generator for Figs. 1-10 (.pdf/.png) & Tables 1-5 (.tex)
├── data/
│   ├── README_DATASETS.md         # Benchmark curation, binarization thresholds & split provenance
│   ├── raw/                       # Curated benchmark CSVs & 5-seed inductive/transductive split manifests
│   └── processed/                 # Pre-processed 3D geometric & PLM tensor manifests (dataset_manifest.json)
├── checkpoints/
│   ├── best.pth                   # Primary best trained PRISM-CPI checkpoint
│   ├── best_BindingDB.pth         # Best trained PRISM-CPI checkpoint on BindingDB
│   ├── best_BioSNAP.pth           # Best trained PRISM-CPI checkpoint on BioSNAP
│   ├── best_PDBbind_2016.pth      # Best trained PRISM-CPI checkpoint on PDBbind 2016
│   └── best_KIBA.pth              # Best trained PRISM-CPI checkpoint on KIBA
├── results/
│   ├── metrics_summary.json       # Complete 5-seed inductive, ablation, transductive & sensitivity metrics
│   ├── shap_attribution_summary.json      # Exact Shapley attribution metrics across all 4 benchmarks
│   ├── docking_network_validation.json    # 30-complex AutoDock Vina energetics, RMSDs & network metrics
│   ├── integrity_report.json      # 800-run completeness & SHA-256 verification report
│   ├── tables/                    # 13 CSV summary tables (Tables 1-5, sensitivity, SHAP, 3D docking, etc.)
│   ├── runs/                      # All 800 individual per-seed experiment run JSON records
│   └── logs/                      # Training curves (training_runs.json.gz) & execution logs
├── docking_3d/                    # 30-complex AutoDock Vina docked poses (.pdb/.sdf/.pdbqt), PyMOL & RDKit outputs
└── figures/                       # Publication-ready vector PDFs, 300-DPI PNGs (Figs. 1-10) & LaTeX tables
```

---

## 3. Environment Setup

```bash
conda create -n prism-cpi python=3.10 -y
conda activate prism-cpi
pip install -r requirements.txt
```

---

## 4. Dataset Preparation & Unpacking

To comply with GitHub's per-file size limits (`< 50 MB` per file) while providing complete data reproducibility directly inside the repository:
- Full master CSVs (`DrugBAN_datasets/bindingdb/full.csv`, `DrugBAN_datasets/biosnap/full.csv`), all 30 five-seed `PDBbind 2016` CSV splits, and representative `Seed 12` inductive splits across all four benchmarks (`BindingDB`, `BioSNAP`, `PDBbind 2016`, `KIBA`) are stored **uncompressed** under `data/raw/` for immediate browser inspection.
- Multi-seed split archives and full training curve logs (`results/logs/training_runs.json.gz`) are stored as compact `.gz` / `.tar.gz` archives (`< 45 MB` each) and can be unpacked in one command:

```bash
python3 scripts/unpack_datasets.py
```

To run preprocessing from raw CSVs (or verify pre-processed tensors):

```bash
# Quick functional test in an isolated temporary directory (completes in ~1 min on CPU/GPU)
bash run_all.sh dry-run

# Full 4-benchmark preprocessing (reads local ./data/raw/ without network calls)
python3 prepare_data.py --mode full --data-dir ./data
```

---

## 5. Training, Evaluation & Reproducing All Manuscript Results

### 5.1 One-Click Full Pipeline Execution (100 Epochs, 5 Seeds: `{12, 14, 16, 18, 20}`)
```bash
# Start full training & evaluation in SSH-safe background daemon mode (with auto-resume)
bash run_all.sh start

# Inspect live progress dashboard or stream logs
bash run_all.sh status
bash run_all.sh logs

# Verify 800-run completeness and package results + checkpoints/best.pth
bash run_all.sh verify
bash run_all.sh package
```

### 5.2 Single Benchmark / Ablation / Baseline Evaluation
```bash
# Train & evaluate PRISM-CPI (Full Model) on BindingDB under cross-cluster inductive split
python3 run.py --dataset BindingDB --scenario inductive --model PRISM-CPI --ablation PRISM_CPI_Full --seed 12 --epochs 100 --mode both

# Run controlled ablation variant (e.g., w/o Module 1 CC-PAE)
python3 run.py --dataset KIBA --scenario inductive --model PRISM-CPI --ablation PRISM_wo_Module1_CC_PAE --seed 12 --epochs 100 --mode both
```

### 5.3 Reproducing Exact SHAP Attribution, 3D AutoDock Vina Redocking & Manuscript Figures
```bash
# 1. Exact 2^8 = 256 coalition Shapley attribution analysis
python3 scripts/run_local_shap_analysis.py

# 2. 30-complex AutoDock Vina v1.2.7 docking, PyMOL 3D pocket rendering & RDKit 2D atom attribution
python3 scripts/postprocess_docking_pymol_rdkit_network.py --render-only

# 3. Regenerate all manuscript figures (Figs. 1-10 .pdf/.png) and LaTeX tables (Tables 1-5 .tex)
python3 scripts/generate_all_figures.py
```

---

## 6. Benchmark Summary (5-Seed Cross-Cluster Inductive Evaluation)

| Benchmark | Compounds | Proteins | Total Pairs | Pos. Ratio | DESTIN AUC | **PRISM-CPI AUC** | DESTIN AUPR | **PRISM-CPI AUPR** | Relative AUPR Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BindingDB** | 14,643 | 2,623 | 49,199 | 42.0% | 0.628 ± 0.018 | **0.689 ± 0.012** | 0.516 ± 0.030 | **0.605 ± 0.020** | **+17.1%** ($p = 1.58 \times 10^{-4}$) |
| **BioSNAP** | 4,505 | 2,181 | 27,464 | 50.4% | 0.805 ± 0.008 | **0.856 ± 0.006** | 0.821 ± 0.010 | **0.869 ± 0.005** | **+5.8%** ($p = 4.30 \times 10^{-5}$) |
| **PDBbind 2016** | 3,227 | 2,410 | 8,106 | 50.0% | 0.724 ± 0.017 | **0.779 ± 0.012** | 0.707 ± 0.017 | **0.775 ± 0.015** | **+9.6%** ($p = 1.00 \times 10^{-6}$) |
| **KIBA** | 2,068 | 229 | 118,254 | 19.2% | 0.624 ± 0.012 | **0.728 ± 0.009** | 0.281 ± 0.011 | **0.447 ± 0.018** | **+59.5%** ($p = 4.00 \times 10^{-6}$) |

---

## 7. Authors & Contact

- **Fang Fang** — School of Information Engineering, Hainan Vocational University of Science and Technology, Haikou 571126, Hainan, China; Faculty of Applied Sciences, Macao Polytechnic University, Macao 999078, Macao, China
- **Wenjuan Jiang** *(Corresponding Author)* — School of Artificial Intelligence, Hainan Normal University, No. 99 Longkun South Road, Haikou 571158, Hainan, China  
  E-mail: [jwj@hainnu.edu.cn](mailto:jwj@hainnu.edu.cn)
- **Rao Zeng** *(Corresponding Author)* — School of Artificial Intelligence & Key Laboratory of Data Science and Smart Education (Ministry of Education), Hainan Normal University, No. 99 Longkun South Road, Haikou 571158, Hainan, China  
  E-mail: [rao.zeng@hainnu.edu.cn](mailto:rao.zeng@hainnu.edu.cn)

For questions, bug reports, or collaboration inquiries, please open a GitHub Issue or contact **Prof. Rao Zeng** (`rao.zeng@hainnu.edu.cn`) and **Prof. Wenjuan Jiang** (`jwj@hainnu.edu.cn`).

---

## 8. License

This project is released under the [MIT License](LICENSE).


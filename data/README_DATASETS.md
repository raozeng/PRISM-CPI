# Benchmark Datasets Overview (BindingDB / BioSNAP / PDBbind 2016 / KIBA)

## 1. Directory Structure

All four compound–protein interaction (CPI) benchmarks are organized under `./data/`:

- **Raw benchmark tables and 5-seed split manifests (`./data/raw/`)**:
  - `DrugBAN_datasets/bindingdb/full.csv` & `DrugBAN_datasets/biosnap/full.csv`: Curated full interaction tables.
  - `GraphBAN_Data/{bindingdb,biosnap,pdb,kiba}/inductive/seed{12,14,16,18,20}/`: Five-seed cross-cluster inductive splits (`source_train`, `target_train`, `target_test`).
  - `GraphBAN_Data/{bindingdb,biosnap,pdb,kiba}/transductive/seed{12,14,16,18,20}/`: Five-seed random transductive splits (`train`, `val`, `test`).
  - `verified_csv_audit.json`: Split statistics and pair counts across all seeds.
  - *Note*: Files larger than 20 MB are stored as `.csv.gz` to stay within GitHub file size limits; run `python3 scripts/unpack_datasets.py` from the repository root to decompress all `.gz` files in place.
- **Preprocessed feature bundles (`./data/processed/`)**:
  - `dataset_manifest.json`: Summary of compound counts, protein counts, active/inactive pairs, feature dimensions, and SHA-256 checksums across all four benchmarks.
  - `KIBA.pt.gz` (and `BindingDB.pt`, `BioSNAP.pt`, `PDBbind_2016.pt` generated via `python3 prepare_data.py --mode full`): Serialized 3D conformer coordinates, 74-D atom descriptors, Uni-Mol embeddings, ProstT5 3Di confidence profiles, SaProt residue representations, and 5-seed split indices.

---

## 2. Benchmark Statistics & Public Provenance

Following the curation and cross-cluster splitting protocols of GraphBAN (*Nature Communications*, 2025) and DESTIN (*Bioinformatics*, 2026), the four benchmarks have the following statistics:

| Dataset | Compounds | Proteins | Active Pairs | Inactive Pairs | Total Pairs | Pos. Ratio | Upstream Public Source |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BindingDB** | 14,643 | 2,623 | 20,674 | 28,525 | 49,199 | 42.0% | [BindingDB](https://www.bindingdb.org/) / [DrugBAN Repository](https://github.com/peizhenbai/DrugBAN/tree/main/datasets/bindingdb) |
| **BioSNAP** | 4,505 | 2,181 | 13,830 | 13,634 | 27,464 | 50.4% | [Stanford SNAP BioSNAP](http://snap.stanford.edu/biodata/) / [DrugBAN Repository](https://github.com/peizhenbai/DrugBAN/tree/main/datasets/biosnap) |
| **PDBbind 2016** | 3,227 | 2,410 | 4,053 | 4,053 | 8,106 | 50.0% | [PDBbind](http://www.pdbbind.org.cn/) / [GraphBAN Repository](https://github.com/hadipourh/GraphBAN) |
| **KIBA** | 2,068 | 229 | 22,729 | 95,525 | 118,254 | 19.2% | [GraphDTA Repository](https://github.com/thinng/GraphDTA/tree/master/data/kiba) / [GraphBAN Repository](https://github.com/hadipourh/GraphBAN) |

---

## 3. Preprocessing Usage

```bash
# Unpack any .gz split archives in place
python3 scripts/unpack_datasets.py

# Build processed feature bundles from ./data/raw/
python3 prepare_data.py --mode full --data-dir ./data
```

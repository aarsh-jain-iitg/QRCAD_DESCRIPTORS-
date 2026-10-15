# dataset/

Put STL files here, one folder per dataset and one sub-folder per class:

```
dataset/
└── <dataset_name>/
    └── <class_name>/
        ├── part_001.stl
        └── part_002.stl
```

- **Class labels come from folder names.** The immediate parent folder of each
  STL is its class. The evaluation (`python main.py --eval`) uses these labels
  for leave-one-out P@k / mAP.
- `src/config.toml` → `[paths].datasets` selects which dataset folders are
  indexed: `["all"]` (every sub-folder) or a list of names.
- Only `*.stl` files (any case) are read; other files are ignored.
- If this folder has no STL files, `python main.py` generates a synthetic
  sample dataset in `dataset/_sample/` (7 classes × 6 parts from trimesh
  primitives, randomly posed and scaled). Delete `_sample/` once you add real
  data, or exclude it via `[paths].datasets`.
- Descriptor vectors are cached per file content in `cache/`; adding, editing
  or deleting STLs only recomputes what changed.

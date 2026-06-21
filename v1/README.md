# Dexter

## 1. Setup

### Install dependencies
```bash
pip install -r requirements.txt
```
Standard packages: PyTorch and Diffusers.

### Download & precompute assets
Run the setup notebook **once** before anything else:
```
notebooks/setup.ipynb
```
This downloads checkpoints and precomputes the tensors the method relies on. Doing it here avoids redundant work and potential errors later.

---

## 2. Running the Method

There are two ways to run Dexter, depending on your goal:

### A) Single run — for understanding & debugging
**Notebook:** `notebooks/run.ipynb`

Use this to run the method once with default (editable) settings. The notebook includes notes on VRAM usage per setting, which is useful if you're working with limited GPU memory.

**Output structure** (written to the output folder):
```
output/
└── <run_name>/
    ├── all_words.txt   ← every word predicted at each step
    └── words.txt       ← only the new best words found
```

> **Note:** There are example configurations in the notebook for **SD1.4**, **SD1.5**, and **SD2.1**. Check those before adapting settings.

---

### B) Systematic runs — for experiments
**Script:** `run_several.py`

Same settings as the notebook, but parametrized for repeated use. This script calls `find_unique_masks()`, which runs the optimization loop until it finds **N distinct words** (capped at **2×N** attempts).

**Output structure** (differs from the notebook):
```
output/
└── <run_name>/
    ├── words_run_0.txt     ← predicted word + loss at each step, for run 0
    ├── words_run_1.txt     ← same for run 1
    ├── ...
    └── found_words.txt     ← one final word per optimization (no duplicates)
```

> You'll notice two folders are created — one is empty. It should be clear from the code why that happens, if you find it annoying you can fix it.

---

### C) Scheduling multiple runs
**Script:** `run_several.sh`

Use this to launch and manage several runs, potentially in parallel across GPUs.

Key things to know:
- **GPU selection** is done via `CUDA_VISIBLE_DEVICES=X` at the top of the script (the method doesn't handle `device` parameters cleanly, so this is the right way to do it).
- **Parallelism:** run the script multiple times with different settings, or append `&` to commands to background them.
- **Parametrize** `run_several.py` with any settings you want to control from bash — checkpoint paths, hyperparameter sweeps via `for` loops, etc.

---

## 3. Visualization

**Notebook:** `notebooks/plot.ipynb`

Generates the plots used in the slides. It reads the discovered words and computes statistics on them.

---

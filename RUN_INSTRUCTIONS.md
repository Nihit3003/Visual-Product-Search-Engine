# End-to-End Run Instructions
## Visual Product Search Engine — From Zero to Demo

---

## Prerequisites

- Kaggle account with GPU accelerator enabled (P100 or T4)
- DeepFashion In-Shop dataset already shared on Kaggle (see Section 1)
- ~3 hours of compute for full training + indexing

---

## Section 1: Kaggle Dataset Setup

### 1.1 Verify the shared dataset

Your friend has shared the dataset. To add it to your notebook:

1. Open Kaggle → **Notebooks** → **New Notebook**
2. Click **+ Add Data** (top right panel) → **Search Datasets**
3. Search: `deepfashion inshop` or ask your friend to share the direct link
4. Click **Add** — it will appear at: `/kaggle/input/<dataset-name>/`

Confirm the structure looks like:
```
/kaggle/input/<dataset-name>/
├── list_eval_partition.txt
├── list_description_inshop.txt
├── list_bbox_inshop.txt
└── img/
    ├── MEN/
    │   ├── Denim/
    │   ├── Jackets_Vests/
    │   └── ...
    └── WOMEN/
```

> **Important**: Note down the exact path — it will be your `DATASET_ROOT`.
> Based on your screenshot, the path is likely:
> `/kaggle/input/deepfashion-inshop/` or `/kaggle/input/<shared-dataset-name>/Dataset/`

---

## Section 2: Upload Project Code to Kaggle

### Option A: GitHub (recommended)

1. Push the entire `visual_search/` folder to a GitHub repo
2. In your Kaggle notebook, add this cell at the top:
```bash
!git clone https://github.com/<your-username>/<your-repo>.git /kaggle/working/visual_search
```

### Option B: Upload directly

1. Zip the `visual_search/` folder
2. Kaggle → Your notebook → **+ Add Data** → **Upload** → upload the zip
3. Unzip in the notebook:
```bash
!unzip /kaggle/input/<your-upload>/visual_search.zip -d /kaggle/working/
```

---

## Section 3: Notebook Setup (run cells in order)

### Cell 1 — Install dependencies

```python
import subprocess, sys

packages = [
    'open-clip-torch',
    'faiss-gpu',
    'transformers>=4.37',
    'ultralytics',
    'accelerate',
    'streamlit',
]
for pkg in packages:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=True)
print('Done.')
```

⏱ **Expected time**: ~3 minutes

### Cell 2 — Set paths

```python
import os, sys

PROJECT_ROOT = '/kaggle/working/visual_search'
sys.path.insert(0, PROJECT_ROOT)

# ⚠️ ADJUST THIS PATH to match your dataset location
DATASET_ROOT = '/kaggle/input/deepfashion-inshop'

OUTPUT_DIR  = '/kaggle/working'
CKPT_DIR    = f'{OUTPUT_DIR}/checkpoints'
INDEX_DIR   = f'{OUTPUT_DIR}/index'
RESULTS_DIR = f'{OUTPUT_DIR}/results'

for d in [CKPT_DIR, INDEX_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Verify files exist
for f in ['list_eval_partition.txt', 'list_description_inshop.txt',
          'list_bbox_inshop.txt', 'img']:
    path = os.path.join(DATASET_ROOT, f)
    print(f'{"✓" if os.path.exists(path) else "✗"} {path}')
```

### Cell 3 — Verify dataset parsing

```python
from src.dataset import parse_eval_partition

splits = parse_eval_partition(f'{DATASET_ROOT}/list_eval_partition.txt')
for k, v in splits.items():
    print(f'{k:8s}: {len(v):,} images')
```

**Expected output:**
```
train   : 25,882 images
query   :  3,997 images
gallery : 12,612 images
```

---

## Section 4: Step A — Build Baseline Index (Condition A)

No training needed. This indexes gallery images using frozen CLIP only.

```bash
!python /kaggle/working/visual_search/scripts/build_index.py \
    --dataset_root {DATASET_ROOT} \
    --index_dir    {INDEX_DIR} \
    --condition    A \
    --alpha        1.0 \
    --batch_size   64 \
    --embed_dim    256
```

⏱ **Expected time**: ~20 minutes (12K gallery images)  
💾 **Output**: `/kaggle/working/index/condition_A_alpha1.0/`

---

## Section 5: Step B — Build Condition B Index (Frozen CLIP + BLIP-2)

This generates captions for every gallery image before indexing.

> ⚠️ BLIP-2 OPT-2.7B requires ~14GB VRAM. If your GPU has <16GB, the model loads in float16 automatically. On a T4 (16GB), this works. On a P100 (16GB), also works.

```bash
!python /kaggle/working/visual_search/scripts/build_index.py \
    --dataset_root {DATASET_ROOT} \
    --index_dir    {INDEX_DIR} \
    --condition    B \
    --alpha        0.6 \
    --batch_size   32 \
    --embed_dim    256
```

⏱ **Expected time**: ~90 minutes (captioning is the bottleneck)  
💾 **Output**: `/kaggle/working/index/condition_B_alpha0.6/`

---

## Section 6: Fine-Tune CLIP (Condition C)

This is the main training step. Run for each seed.

### Single-seed training

```bash
!python /kaggle/working/visual_search/scripts/train_clip.py \
    --dataset_root    {DATASET_ROOT} \
    --output_dir      {CKPT_DIR}/seed_42 \
    --epochs          10 \
    --batch_size      128 \
    --lr              1e-4 \
    --unfreeze_last_n 4 \
    --embed_dim       256 \
    --seed            42 \
    --eval_every      2
```

⏱ **Expected time**: ~60 minutes (10 epochs, GPU accelerated)  
💾 **Output**: `/kaggle/working/checkpoints/seed_42/clip_finetuned_best.pt`

### Multi-seed training (for ablation)

```python
SEEDS = [42, 2024, 1337, 7]  # Replace with team roll numbers

for seed in SEEDS:
    print(f'\n===== Seed {seed} =====')
    os.system(f"""
    python /kaggle/working/visual_search/scripts/train_clip.py \
        --dataset_root    {DATASET_ROOT} \
        --output_dir      {CKPT_DIR}/seed_{seed} \
        --epochs          10 \
        --batch_size      128 \
        --lr              1e-4 \
        --seed            {seed}
    """)
```

⏱ **Expected time**: ~4 hours for 4 seeds

**Tip for Kaggle time limits**: Kaggle sessions are 12 hours. Save checkpoints after every seed:
```bash
!cp -r {CKPT_DIR} /kaggle/working/checkpoints_backup
```

---

## Section 7: Build Condition C Index

```python
CKPT_PATH = f'{CKPT_DIR}/seed_42/clip_finetuned_best.pt'
```

```bash
!python /kaggle/working/visual_search/scripts/build_index.py \
    --dataset_root {DATASET_ROOT} \
    --ckpt_path    {CKPT_PATH} \
    --index_dir    {INDEX_DIR} \
    --condition    C \
    --alpha        0.6 \
    --batch_size   64 \
    --embed_dim    256
```

⏱ **Expected time**: ~30 minutes  
💾 **Output**: `/kaggle/working/index/condition_C_alpha0.6/`

---

## Section 8: Run Full Evaluation

This runs Recall@K, NDCG@K, mAP@K for all three conditions across all seeds.

```bash
!python /kaggle/working/visual_search/scripts/evaluate.py \
    --dataset_root {DATASET_ROOT} \
    --index_base   {INDEX_DIR} \
    --ckpt_path    {CKPT_PATH} \
    --output_dir   {RESULTS_DIR} \
    --seeds        42 2024 1337 7 \
    --batch_size   64 \
    --alpha_B      0.6 \
    --alpha_C      0.6 \
    --use_itm
```

⏱ **Expected time**: ~60 minutes  
💾 **Output**: `/kaggle/working/results/ablation_results.json`

### Display results table

```python
import json, pandas as pd

with open(f'{RESULTS_DIR}/ablation_results.json') as f:
    res = json.load(f)

rows = []
for cond, metrics in res.items():
    row = {'Condition': cond.replace('condition_', '')}
    for metric, vals in metrics.items():
        row[metric] = f"{vals['mean']:.4f} ± {vals['std']:.4f}"
    rows.append(row)

df = pd.DataFrame(rows).set_index('Condition')
print(df.to_string())
df
```

**Expected output format:**
```
          Recall@5     Recall@10    Recall@15    NDCG@5    ...   mAP@15
A        0.6123±0.0041  0.7012±0.0038  ...
B        0.6891±0.0029  0.7634±0.0025  ...
C        0.7421±0.0033  0.8102±0.0021  ...
```

---

## Section 9: Launch Streamlit Demo

### Option A: Via ngrok (Kaggle)

```python
!pip install -q pyngrok
from pyngrok import ngrok
import subprocess, threading, time

NGROK_TOKEN = "your_ngrok_auth_token_here"  # Get free token at ngrok.com
ngrok.set_auth_token(NGROK_TOKEN)

def run_app():
    subprocess.run([
        'streamlit', 'run',
        '/kaggle/working/visual_search/app/demo.py',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--',
        '--dataset_root', DATASET_ROOT,
        '--index_dir', f'{INDEX_DIR}/condition_C_alpha0.6',
        '--ckpt_path', CKPT_PATH,
        '--alpha', '0.6',
        '--top_k', '12',
    ])

t = threading.Thread(target=run_app, daemon=True)
t.start()
time.sleep(8)
public_url = ngrok.connect(8501)
print(f'🌍 Demo URL: {public_url}')
```

### Option B: Run locally (after downloading outputs)

```bash
# 1. Download checkpoints and index from Kaggle to your machine
# 2. Install requirements
pip install -r requirements.txt

# 3. Run the demo
streamlit run app/demo.py -- \
    --dataset_root /path/to/deepfashion \
    --index_dir    /path/to/index/condition_C_alpha0.6 \
    --ckpt_path    /path/to/clip_finetuned_best.pt
```

---

## Section 10: Demo Script for Batch Evaluation (Custom Query Folder)

Given a folder of your own query images:

```bash
# Create a folder with test images
mkdir /kaggle/working/my_queries
# Upload some clothing images into it

# Run evaluation on custom query images
python /kaggle/working/visual_search/scripts/evaluate.py \
    --dataset_root {DATASET_ROOT} \
    --index_base   {INDEX_DIR} \
    --ckpt_path    {CKPT_PATH} \
    --output_dir   {RESULTS_DIR}/custom \
    --query_folder /kaggle/working/my_queries \
    --single_run
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'open_clip'`
```bash
pip install open-clip-torch
```

### `ModuleNotFoundError: No module named 'faiss'`
```bash
pip install faiss-gpu   # with GPU
pip install faiss-cpu   # CPU only
```

### BLIP-2 CUDA out of memory
Reduce batch size: `--batch_size 16`  
Or use CPU: models will load slower but work.

### YOLO weights not found
```bash
# YOLOv8n auto-downloads on first use. If no internet:
# Download yolov8n.pt from https://github.com/ultralytics/assets/releases
# Place at /kaggle/working/yolov8n.pt and pass --yolo_weights /kaggle/working/yolov8n.pt
```

### Dataset path not found
Check your dataset input path:
```python
import os
for f in os.listdir('/kaggle/input'):
    print(f)
```
This shows all available datasets. Update `DATASET_ROOT` accordingly.

### Kaggle session expires mid-training
- Use Kaggle's "Save & Run All" button which runs in the background
- Alternatively, use `--resume /path/to/last_checkpoint.pt` to continue

---

## Quick Summary Checklist

- [ ] Dataset added to Kaggle notebook
- [ ] Project code uploaded / cloned
- [ ] `DATASET_ROOT` path verified
- [ ] Dependencies installed
- [ ] Condition A index built
- [ ] CLIP fine-tuned (Condition C) — at least 1 seed
- [ ] Condition B index built
- [ ] Condition C index built
- [ ] Ablation evaluation run
- [ ] Results JSON generated and table displayed
- [ ] Streamlit demo launched with ngrok URL

---

*Total estimated compute time on Kaggle T4/P100:*  
*Training 4 seeds + all indexing + evaluation ≈ 6–7 hours*  
*Single seed demo (enough to show working system) ≈ 2–3 hours*

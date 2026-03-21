# 🔮 PRISM — Plan of Action

**For the GPU team: step-by-step guide to train, evaluate, and finalize PRISM.**

> ⏱️ **Estimated total time:** ~1-2 hours (mostly training time)

---

## ✅ What's Already Done (by Lokaksha)

- [x] Dataset analyzed (10 scenes, 404 images, 1600×900)
- [x] Full codebase written (7 Python modules, 4,300 lines)
- [x] Model architecture verified (1.86M student / 4.56M teacher)
- [x] 2-epoch smoke test passed on CPU
- [x] Pushed to GitHub

---

## 🎯 What You Need To Do

### STEP 1 — Setup (~5 min)

```bash
# Clone the repo
git clone https://github.com/Lokaksha25/PRISM-Path-Recognition-with-Intelligent-Segmentation-Model-.git
cd PRISM-Path-Recognition-with-Intelligent-Segmentation-Model-

# Install dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

> ⚠️ If CUDA isn't detected, install PyTorch with CUDA from https://pytorch.org/get-started/locally/

---

### STEP 2 — Dataset Setup (~10 min)

Copy the **nuScenes v1.0-mini** dataset into the project folder. You need these folders:
```
Project Root/
├── v1.0-mini/        ← metadata JSONs (scene.json, sample.json, etc.)
├── samples/          ← camera images
│   └── CAM_FRONT/    ← 404 front camera images
├── sweeps/           ← sweep data
└── maps/             ← BEV map images
```

**Option A:** Copy from Lokaksha's machine (he has it extracted already)  
**Option B:** Extract from the `.tgz` file:
```bash
tar -xzf v1.0-mini.tgz
```

---

### STEP 3 — Generate Drivable Masks (~3 min)

```bash
python generate_masks.py --dataroot ./ --output_dir masks --visualize 5
```

**Check:** Open `masks/visualizations/` and verify the 5 sample images look reasonable — green overlay should cover the road area.

**Expected output:**
- 404 mask PNGs in `masks/`
- 5 visualization samples in `masks/visualizations/`

---

### STEP 4 — Train Student Model (~15-20 min on GPU)

```bash
python train.py --dataroot ./ --mask_dir masks --epochs 50 --batch_size 16 --lr 1e-3
```

**What to watch for:**
- Loss should decrease steadily
- mIoU should climb (0.3 → 0.5 → 0.7+)
- Best checkpoint auto-saves to `output/best_model.pth`
- Every 10 epochs it prints a detailed report with FPS

**If you get CUDA out of memory:** reduce batch size:
```bash
python train.py --epochs 50 --batch_size 8
```

---

### STEP 5 — Train Teacher Model (~20-25 min on GPU)

```bash
python train.py --epochs 50 --batch_size 16 --train_teacher --save_name teacher_best.pth
```

This trains the larger 4.56M param model — needed for knowledge distillation.

---

### STEP 6 — Knowledge Distillation (~15-20 min on GPU)

```bash
python train.py --epochs 50 --batch_size 16 --distill --teacher_weights output/teacher_best.pth --save_name distilled_model.pth
```

This trains the student model using soft targets from the teacher — should give a mIoU boost.

---

### STEP 7 — Evaluate Best Model (~2 min)

```bash
# With TTA + boundary refinement (best results)
python evaluate.py --weights output/best_model.pth --use_tta --use_boundary_refinement

# Also evaluate the distilled model
python evaluate.py --weights output/distilled_model.pth --use_tta --use_boundary_refinement --output_dir eval_distilled
```

**This produces:**
- Final mIoU, confusion matrix
- Per-scene breakdown (best/worst 3 scenes)
- 10 overlay visualizations in `eval_output/visualizations/`
- `eval_output/confusion_matrix.png`

---

### STEP 8 — ONNX Export + Benchmarks (~2 min)

```bash
python inference.py --export_onnx --benchmark --quantize --weights output/best_model.pth
```

**This produces:**
- `model.onnx` — exported model
- FPS comparison: PyTorch vs ONNX vs Quantized
- Model size comparison

---

### STEP 9 — Demo Video (~1 min)

```bash
python inference.py --demo_video --weights output/best_model.pth --dataroot ./
```

Creates `inference_output/demo_video.mp4` — segmentation overlay running over a scene sequence.

---

### STEP 10 — Collect Deliverables

After all steps, gather these files and send them back:

| File | Location |
|---|---|
| Best student weights | `output/best_model.pth` |
| Teacher weights | `output/teacher_best.pth` |
| Distilled weights | `output/distilled_model.pth` |
| Training curves | `output/training_curves.png` |
| Training history | `output/training_history.json` |
| ONNX model | `model.onnx` |
| Metrics JSON | `eval_output/metrics.json` |
| Confusion matrix | `eval_output/confusion_matrix.png` |
| 10 visualizations | `eval_output/visualizations/` |
| Demo video | `inference_output/demo_video.mp4` |

---

## 🔥 Quick Copy-Paste (All Commands)

```bash
# Run these in order:
pip install -r requirements.txt
python generate_masks.py --dataroot ./ --output_dir masks --visualize 5
python train.py --epochs 50 --batch_size 16
python train.py --epochs 50 --batch_size 16 --train_teacher --save_name teacher_best.pth
python train.py --epochs 50 --batch_size 16 --distill --teacher_weights output/teacher_best.pth --save_name distilled_model.pth
python evaluate.py --weights output/best_model.pth --use_tta --use_boundary_refinement
python inference.py --export_onnx --benchmark --quantize --weights output/best_model.pth
python inference.py --demo_video --weights output/best_model.pth --dataroot ./
```

---

## 🆘 Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Use `--batch_size 8` or `--batch_size 4` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: masks/` | Run `generate_masks.py` first |
| `No CAM_FRONT images` | Make sure `samples/CAM_FRONT/` folder exists with 404 `.jpg` files |
| Training loss not decreasing | Normal for first 2-3 epochs, should improve after epoch 5 |

---

*Questions? Text Lokaksha — he built the entire codebase and knows every line. 💪*

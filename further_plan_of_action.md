# PRISM: Further Plan of Action

Following the successful merging of the advanced `PRISMLoss` function branch, the foundational code is officially complete. The `PLAN_OF_ACTION.md` indicates that data analysis, codebase authoring, model verification, and smoke tests are formally checked off. 

The immediate next steps transition the project from the **development phase** to the **GPU execution and evaluation phase**.

---

## Phase 1: Environment & Data Preparation (GPU Server)
Since previous tests ran on CPU, the execution now shifts to the GPU machine for full-scale training.

1. **Setup Workspace**: Clone the latest repository (which now includes the improved `PRISMLoss`) on the target GPU machine and install requirements.
2. **Dataset Extraction**: Download and extract the `nuScenes v1.0-mini` dataset into the project directory.
3. **Mask Generation**: Run `generate_masks.py` to pre-generate the 404 ground-truth drivable area masks. This step converts NuScenes vector data into training-ready PNG overlays.

## Phase 2: Core Training Strategy
This phase relies heavily on the latest branch merge. The new `PRISMLoss` will automatically penalize blurry boundaries and false negatives.

1. **Student Model Baseline (50 epochs)**: 
   - Train the lightweight (1.86M param) student model from scratch.
   - Monitor the `mIoU` score (expected to rise faster thanks to `BFABoundaryLoss`).
2. **Teacher Model Training (50 epochs)**: 
   - Train the heavy (4.56M param) `LiteSegTeacher` model. The larger capacity will generate highly accurate, generalized soft-labels for the student.
3. **Knowledge Distillation (50 epochs)**:
   - Distill knowledge from the teacher back to the student. 
   - The updated `DistillationLoss` defaults to using the new `PRISMLoss` for target supervision while simultaneously learning from the teacher's soft probabilities.

## Phase 3: Final Evaluation & Benchmarking
Once the optimal distilled student model is isolated (`distilled_model.pth`), the pipeline must be proven effective and performant.

1. **Metric Evaluation**: Run `evaluate.py` enabling Test-Time Augmentation (TTA) and boundary refinement. This generates the final confusion matrix, mIoU scores, and hard-case visualizations.
2. **Hardware Benchmarking**: Export the model to ONNX (`inference.py --export_onnx --benchmark`). Compare latency (FPS) and artifact size between raw PyTorch, pure ONNX, and Quantized ONNX formats.
3. **Visual Proof (Demo Video)**: Generate a continuous video overlay segmenting scenes in real-time (`inference.py --demo_video`). 

## Phase 4: Delivery
Gather all artifacts (`best_model.pth`, `teacher_best.pth`, `distilled_model.pth`, `metrics.json`, benchmarking graphs, and `demo_video.mp4`) for final hackathon submission.

---
**Summary Check**: The codebase is locked in and theoretically optimized for road boundaries. The entire remaining workload is execution, benchmarking, and packaging.

$ErrorActionPreference = "Stop"

Write-Host "--- PRISM Full Retraining Script ---"
Write-Host "Installing requirements..."
pip install -r requirements.txt

Write-Host "1/8 Generating masks..."
python generate_masks.py --dataroot ./ --output_dir masks --visualize 5

Write-Host "2/8 Training student model..."
python train.py --epochs 50 --batch_size 16

Write-Host "3/8 Training teacher model..."
python train.py --epochs 50 --batch_size 16 --train_teacher --save_name teacher_best.pth

Write-Host "4/8 Knowledge distillation..."
python train.py --epochs 50 --batch_size 16 --distill --teacher_weights output/teacher_best.pth --save_name distilled_model.pth

Write-Host "5/8 Evaluating student model..."
python evaluate.py --weights output/best_model.pth --use_tta --use_boundary_refinement
python evaluate.py --weights output/distilled_model.pth --use_tta --use_boundary_refinement --output_dir eval_distilled

Write-Host "6/8 Exporting ONNX & Benchmarking..."
python inference.py --export_onnx --benchmark --quantize --weights output/best_model.pth

Write-Host "7/8 Generating Demo Video..."
python inference.py --demo_video --weights output/best_model.pth --dataroot ./

Write-Host "8/8 Zipping Deliverables..."
if (Test-Path "prism_deliverables.zip") { Remove-Item "prism_deliverables.zip" }
Compress-Archive -Path output/best_model.pth, output/teacher_best.pth, output/distilled_model.pth, output/training_curves.png, output/training_history.json, model.onnx, eval_output/metrics.json, eval_output/confusion_matrix.png, eval_output/visualizations, inference_output/demo_video.mp4 -DestinationPath prism_deliverables.zip -Force

Write-Host "--- ALL DONE ---"

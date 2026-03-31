import json
h = json.load(open('output/training_history.json'))
n = len(h['train_loss'])
print("=== STUDENT TRAINING RESULTS ===")
print(f"Epochs completed: {n} (early stopping)")
print(f"Final - Train Loss: {h['train_loss'][-1]:.4f} | Val Loss: {h['val_loss'][-1]:.4f}")
print(f"Final - Train mIoU: {h['train_miou'][-1]:.4f} | Val mIoU: {h['val_miou'][-1]:.4f}")
print(f"Val Precision: {h['val_precision'][-1]:.4f}")
print(f"Val Recall:    {h['val_recall'][-1]:.4f}")
print(f"Val FPR:       {h['val_fpr'][-1]:.4f}")
print(f"Val F1:        {h['val_f1'][-1]:.4f}")
best_miou = max(h['val_miou'])
best_ep = h['val_miou'].index(best_miou) + 1
print(f"Best Val mIoU: {best_miou:.4f} at epoch {best_ep}")

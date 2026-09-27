import os
import time
import json
import torch
import evaluate
import numpy as np

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Computes Accuracy and F1 Score."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1": f1}

def get_parameter_stats(model):
    """Calculates total and trainable parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable_params / total_params
    return trainable_params, total_params, pct

class PerformanceTracker:
    """Tracks training time and peak GPU memory."""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self):
        self.end_time = time.time()
        peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        peak_mb = peak_bytes / (1024 ** 2)
        duration_sec = self.end_time - self.start_time
        return duration_sec, peak_mb

def save_experiment_results(results_dict, output_path):
    """Saves metrics to JSON for Phase 6 aggregation."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print(f"Metrics saved to {output_path}")

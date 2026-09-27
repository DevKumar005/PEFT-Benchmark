import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from utils import compute_metrics, get_parameter_stats, PerformanceTracker, save_experiment_results

def run_full_finetuning(
    data_dir="data_cache",
    output_dir="results/full_ft",
    epochs=3,
    batch_size=16,
    lr=2e-5,
    seed=42
):
    set_seed(seed)
    print("\n--- Starting Full Fine-Tuning Pipeline ---")

    # 1. Load cached splits
    train_data = load_from_disk(os.path.join(data_dir, "train"))
    val_data = load_from_disk(os.path.join(data_dir, "val"))
    test_data = load_from_disk(os.path.join(data_dir, "test"))

    # 2. Load model (all parameters trainable)
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    trainable_p, total_p, pct = get_parameter_stats(model)
    print(f"Total Params: {total_p:,} | Trainable: {trainable_p:,} ({pct:.2f}%)")

    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),  # Fast standard mixed precision on T4
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )

    # 4. Measure Training Duration and Peak Memory
    tracker = PerformanceTracker()
    tracker.start()
    train_result = trainer.train()
    duration_sec, peak_mem_mb = tracker.stop()

    print(f"\nTraining Complete!")
    print(f"Wall-clock Time: {duration_sec:.2f}s ({duration_sec / 60:.2f} min)")
    print(f"Peak GPU VRAM: {peak_mem_mb:.2f} MB")

    # 5. Evaluate on Held-out Test Set
    print("\nEvaluating on Test Split...")
    test_metrics = trainer.evaluate(test_data)
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['eval_f1']:.4f}")

    # 6. Extract loss history
    loss_history = [
        {"step": log["step"], "loss": log["loss"]}
        for log in trainer.state.log_history
        if "loss" in log
    ]

    results = {
        "technique": "Full Fine-Tuning",
        "total_params": total_p,
        "trainable_params": trainable_p,
        "trainable_percent": pct,
        "training_time_sec": duration_sec,
        "peak_memory_mb": peak_mem_mb,
        "test_accuracy": test_metrics["eval_accuracy"],
        "test_f1": test_metrics["eval_f1"],
        "loss_history": loss_history
    }

    save_experiment_results(results, os.path.join(output_dir, "metrics.json"))
    return results

if __name__ == "__main__":
    run_full_finetuning()

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from utils import compute_metrics, get_parameter_stats, PerformanceTracker, save_experiment_results

def run_lora_finetuning(
    data_dir="data_cache",
    output_dir="results/lora",
    epochs=3,
    batch_size=16,
    lr=5e-4,  # LoRA typically benefits from a higher learning rate than full FT
    seed=42
):
    set_seed(seed)
    print("\n--- Starting LoRA Fine-Tuning Pipeline ---")

    # 1. Load dataset splits
    train_data = load_from_disk(os.path.join(data_dir, "train"))
    val_data = load_from_disk(os.path.join(data_dir, "val"))
    test_data = load_from_disk(os.path.join(data_dir, "test"))

    # 2. Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    # 3. Configure LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
        bias="none"
    )

    model = get_peft_model(base_model, lora_config)
    trainable_p, total_p, pct = get_parameter_stats(model)
    print(f"Total Params: {total_p:,} | Trainable: {trainable_p:,} ({pct:.2f}%)")

    # 4. Training Arguments
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
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics
    )

    # 5. Measure Duration and Peak VRAM
    tracker = PerformanceTracker()
    tracker.start()
    train_result = trainer.train()
    duration_sec, peak_mem_mb = tracker.stop()

    print(f"\nTraining Complete!")
    print(f"Wall-clock Time: {duration_sec:.2f}s ({duration_sec / 60:.2f} min)")
    print(f"Peak GPU VRAM: {peak_mem_mb:.2f} MB")

    # 6. Evaluate on Test Set
    print("\nEvaluating on Test Split...")
    test_metrics = trainer.evaluate(test_data)
    print(f"Test Accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['eval_f1']:.4f}")

    # 7. Extract loss history
    loss_history = [
        {"step": log["step"], "loss": log["loss"]}
        for log in trainer.state.log_history
        if "loss" in log
    ]

    results = {
        "technique": "LoRA",
        "total_params": total_p,
        "trainable_params": trainable_p,
        "trainable_percent": pct,
        "training_time_sec": duration_sec,
        "peak_memory_mb": peak_mem_mb,
        "test_accuracy": test_metrics["eval_accuracy"],
        "test_f1": test_metrics["eval_f1"],
        "loss_history": loss_history
    }

    # Save metrics and adapter weights
    save_experiment_results(results, os.path.join(output_dir, "metrics.json"))
    model.save_pretrained(os.path.join(output_dir, "adapter_weights"))
    return results

if __name__ == "__main__":
    run_lora_finetuning()

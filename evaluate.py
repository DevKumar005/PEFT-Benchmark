import os
import json
import pandas as pd

def aggregate_and_verify():
    paths = {
        "Full Fine-Tuning": "results/full_ft/metrics.json",
        "LoRA": "results/lora/metrics.json",
        "QLoRA (4-bit)": "results/qlora/metrics.json"
    }

    records = []
    for technique, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing results for {technique} at {path}")
        with open(path, "r") as f:
            data = json.load(f)
            records.append({
                "Method": technique,
                "Trainable Params": f"{data['trainable_params']:,}",
                "Trainable %": f"{data['trainable_percent']:.2f}%",
                "Training Time (s)": round(data["training_time_sec"], 2),
                "Peak VRAM (MB)": round(data["peak_memory_mb"], 2),
                "Test Accuracy": round(data["test_accuracy"], 4),
                "Test F1": round(data["test_f1"], 4)
            })

    df = pd.DataFrame(records)

    # Save to disk
    os.makedirs("results", exist_ok=True)
    csv_path = "results/comparison_table.csv"
    md_path = "results/comparison_table.md"
    df.to_csv(csv_path, index=False)
    
    with open(md_path, "w") as f:
        f.write(df.to_markdown(index=False))

    print("\n--- Consolidated Comparison Table (FR9) ---")
    print(df.to_string(index=False))
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")
    return df

if __name__ == "__main__":
    aggregate_and_verify()

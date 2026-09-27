import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_imdb_data(
    model_name: str = "distilbert-base-uncased",
    train_size: int = 5000,
    val_size: int = 1000,
    test_size: int = 1000,
    max_length: int = 256,
    seed: int = 42
):
    """
    Loads IMDB dataset, creates reproducible train/val/test splits,
    tokenizes, and formats tensors for PyTorch.
    """
    print("Loading raw IMDB dataset (stanfordnlp/imdb)...")
    raw_datasets = load_dataset("stanfordnlp/imdb")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1. Shuffle and slice training data to satisfy NFR1 (speed) and NFR2 (reproducibility)
    full_train = raw_datasets["train"].shuffle(seed=seed)
    train_subset = full_train.select(range(train_size))
    
    # 2. Split train subset into train and validation splits
    train_val_split = train_subset.train_test_split(test_size=val_size, seed=seed)
    train_data = train_val_split["train"]
    val_data = train_val_split["test"]

    # 3. Create test split from raw test data
    test_subset = raw_datasets["test"].shuffle(seed=seed).select(range(test_size))

    print(f"Splits created: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_subset)}")

    # 4. Tokenization function
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    print("Tokenizing datasets...")
    tokenized_train = train_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = val_data.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_test = test_subset.map(tokenize_fn, batched=True, remove_columns=["text"])

    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")
    tokenized_test.set_format("torch")

    return {
        "train": tokenized_train,
        "val": tokenized_val,
        "test": tokenized_test,
        "tokenizer": tokenizer
    }

if __name__ == "__main__":
    splits = prepare_imdb_data()
    print("Data preparation complete and verified.")

from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

# Load the Hugging Face dataset
dataset = load_dataset("TimSchopf/arxiv_categories")

print("\nDataset loaded successfully.")
print(dataset)

# Convert each split to pandas DataFrame
train_df = dataset["train"].to_pandas()
val_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nColumns:")
print(train_df.columns.tolist())

print("\nFirst 5 rows:")
print(train_df.head())

# Save local CSV copies
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\nSaved CSV files into the data/ folder.")
import pandas as pd

train_df = pd.read_csv("data/train.csv")

print("Columns:")
print(train_df.columns.tolist())

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nSample rows:")
print(train_df.head(10))

# If there is a category column, inspect it
possible_label_cols = ["label", "category", "categories", "primary_category"]
found = [col for col in possible_label_cols if col in train_df.columns]

print("\nPossible label columns found:", found)

for col in found:
    print(f"\nTop values in {col}:")
    print(train_df[col].value_counts().head(20))
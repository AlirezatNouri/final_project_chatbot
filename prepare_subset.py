import pandas as pd
import re

# Load splits
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

# Combine them for preprocessing, then we can split again later if needed
df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print("Original shape:", df.shape)

# Extract all CS category codes from the category string
def extract_cs_labels(cat_str):
    if pd.isna(cat_str):
        return []
    return re.findall(r"(cs\.[A-Z]{2})", str(cat_str))

df["cs_labels"] = df["categories"].apply(extract_cs_labels)

# Keep rows that have at least one CS label
df = df[df["cs_labels"].map(len) > 0].copy()

# Use the first CS label as the primary label
df["primary_cs_label"] = df["cs_labels"].apply(lambda x: x[0])

print("After keeping CS papers:", df.shape)

print("\nTop CS labels:")
print(df["primary_cs_label"].value_counts().head(20))

# Choose final target categories
target_labels = ["cs.CL", "cs.LG", "cs.CV", "cs.DB", "cs.IR"]

subset_df = df[df["primary_cs_label"].isin(target_labels)].copy()

print("\nSubset shape:", subset_df.shape)
print("\nSubset class distribution:")
print(subset_df["primary_cs_label"].value_counts())

# Keep only useful columns
subset_df = subset_df[["id", "title", "abstract", "primary_cs_label", "creation_date"]]

# Rename the label column
subset_df = subset_df.rename(columns={"primary_cs_label": "label"})

# Save
subset_df.to_csv("data/research_papers_subset.csv", index=False)

print("\nSaved: data/research_papers_subset.csv")
print("\nSample:")
print(subset_df.head())
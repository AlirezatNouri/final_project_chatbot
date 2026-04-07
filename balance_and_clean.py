import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

SEED = 42
TARGET_PER_CLASS = 350

df = pd.read_csv("data/research_papers_subset.csv")

print("Original subset shape:", df.shape)
print("\nOriginal class distribution:")
print(df["label"].value_counts())

# Sample equal number from each class
balanced_df = (
    df.groupby("label", as_index=False)
      .sample(n=TARGET_PER_CLASS, random_state=SEED)
      .reset_index(drop=True)
)

print("\nBalanced class distribution:")
print(balanced_df["label"].value_counts())
print("\nBalanced shape:", balanced_df.shape)

# Combine title and abstract
balanced_df["text"] = balanced_df["title"].fillna("") + " " + balanced_df["abstract"].fillna("")

# Clean text
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

balanced_df["clean_text"] = balanced_df["text"].apply(clean_text)

# Keep only useful columns
balanced_df = balanced_df[["id", "title", "abstract", "label", "creation_date", "text", "clean_text"]]

# Split into train/val/test
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=0.3,
    stratify=balanced_df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED
)

print("\nTrain shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain class distribution:")
print(train_df["label"].value_counts())

# Save files
balanced_df.to_csv("data/research_papers_balanced.csv", index=False)
train_df.to_csv("data/train_balanced.csv", index=False)
val_df.to_csv("data/val_balanced.csv", index=False)
test_df.to_csv("data/test_balanced.csv", index=False)

print("\nSaved:")
print("- data/research_papers_balanced.csv")
print("- data/train_balanced.csv")
print("- data/val_balanced.csv")
print("- data/test_balanced.csv")

print("\nSample cleaned rows:")
print(balanced_df[["title", "label", "clean_text"]].head())
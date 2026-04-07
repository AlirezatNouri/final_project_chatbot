import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =========================
# 1. Load data
# =========================
train_df = pd.read_csv("data/train_balanced.csv")
val_df = pd.read_csv("data/val_balanced.csv")
test_df = pd.read_csv("data/test_balanced.csv")

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# =========================
# 2. Extract text and labels
# =========================
X_train_text = train_df["clean_text"]
y_train = train_df["label"]

X_val_text = val_df["clean_text"]
y_val = val_df["label"]

X_test_text = test_df["clean_text"]
y_test = test_df["label"]

# =========================
# 3. TF-IDF feature engineering
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)
X_test = vectorizer.transform(X_test_text)

print("\nTF-IDF feature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# =========================
# 4. Define models
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=42)
}

results = []

# =========================
# 5. Train and evaluate
# =========================
for model_name, model in models.items():
    print(f"\n==============================")
    print(f"Training: {model_name}")
    print(f"==============================")

    model.fit(X_train, y_train)

    # Validation predictions
    y_val_pred = model.predict(X_val)

    # Test predictions
    y_test_pred = model.predict(X_test)

    # Validation metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="weighted"
    )

    # Test metrics
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="weighted"
    )

    results.append({
        "Model": model_name,
        "Val Accuracy": val_acc,
        "Val Precision": val_precision,
        "Val Recall": val_recall,
        "Val F1": val_f1,
        "Test Accuracy": test_acc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Test F1": test_f1
    })

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred))

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))

    # Save confusion matrix for test set
    cm = confusion_matrix(y_test, y_test_pred, labels=sorted(y_test.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    filename = f"results/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix: {filename}")

# =========================
# 6. Save results
# =========================
results_df = pd.DataFrame(results)
print("\nFinal comparison table:")
print(results_df)

results_df.to_csv("results/classification_results.csv", index=False)
print("\nSaved: results/classification_results.csv")

# =========================
# 7. Plot comparison chart
# =========================
plt.figure(figsize=(10, 6))
plt.bar(results_df["Model"], results_df["Test F1"])
plt.title("Test F1 Score Comparison of Classifiers")
plt.ylabel("F1 Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/classifier_f1_comparison.png")
plt.close()

print("Saved: results/classifier_f1_comparison.png")
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# =========================
# 1. Load data
# =========================
train_df = pd.read_csv("data/train_balanced.csv")
val_df = pd.read_csv("data/val_balanced.csv")
test_df = pd.read_csv("data/test_balanced.csv")
clustered_df = pd.read_csv("results/papers_with_clusters.csv")

# Merge all balanced papers for retrieval
papers_df = clustered_df.copy()

print("Papers loaded:", papers_df.shape)

# =========================
# 2. Prepare training data
# =========================
X_train_text = train_df["clean_text"]
y_train = train_df["label"]

# =========================
# 3. TF-IDF vectorizer
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_papers = vectorizer.transform(papers_df["clean_text"])

# =========================
# 4. Train best classifier
# =========================
clf = MultinomialNB()
clf.fit(X_train, y_train)

print("Naive Bayes classifier trained.")

# =========================
# 5. Text cleaning function
# =========================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# 6. Recommendation function
# =========================
def recommend_papers(query, top_k=5):
    clean_query = clean_text(query)
    query_vec = vectorizer.transform([clean_query])

    # Predict topic
    predicted_label = clf.predict(query_vec)[0]
    predicted_probs = clf.predict_proba(query_vec)[0]
    class_confidence = max(predicted_probs)

    # Similarity with all papers
    sims = cosine_similarity(query_vec, X_papers).flatten()

    # Base recommendation dataframe
    results = papers_df.copy()
    results["similarity"] = sims

    # Boost score for same predicted label
    results["label_boost"] = (results["label"] == predicted_label).astype(int) * 0.15

    # Final score
    results["final_score"] = results["similarity"] + results["label_boost"]

    # Sort and take top results
    top_results = results.sort_values("final_score", ascending=False).head(top_k)

    # Build explanations
    recommendations = []
    for _, row in top_results.iterrows():
        summary = row["abstract"][:350] + "..." if len(row["abstract"]) > 350 else row["abstract"]
        explanation = (
            f"Recommended because it is highly similar to your query and matches the predicted topic "
            f"'{predicted_label}'."
        )
        recommendations.append({
            "title": row["title"],
            "label": row["label"],
            "cluster": row["kmeans_cluster"],
            "similarity": round(row["similarity"], 4),
            "final_score": round(row["final_score"], 4),
            "summary": summary,
            "explanation": explanation
        })

    return predicted_label, round(class_confidence, 4), recommendations

# =========================
# 7. Demo queries
# =========================
demo_queries = [
    "I want papers about NLP translation models",
    "recommend papers on recommender systems",
    "find papers about image segmentation and object detection",
    "show me papers on databases and query optimization",
    "papers about deep learning and neural networks"
]

for q in demo_queries:
    print("\n" + "="*80)
    print("QUERY:", q)

    pred_label, conf, recs = recommend_papers(q, top_k=3)

    print("Predicted topic:", pred_label)
    print("Confidence:", conf)

    for i, rec in enumerate(recs, start=1):
        print(f"\nRecommendation {i}:")
        print("Title:", rec["title"])
        print("Label:", rec["label"])
        print("Cluster:", rec["cluster"])
        print("Similarity:", rec["similarity"])
        print("Final Score:", rec["final_score"])
        print("Explanation:", rec["explanation"])
        print("Summary:", rec["summary"][:200], "...")
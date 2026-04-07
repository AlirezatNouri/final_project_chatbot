import pandas as pd
import re
import nltk

from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

app = FastAPI(title="Research Paper Recommender API")

# =========================
# Load data
# =========================
train_df = pd.read_csv("data/train_balanced.csv")
papers_df = pd.read_csv("results/papers_with_clusters.csv")

# =========================
# Build vectorizer + classifier
# =========================
X_train_text = train_df["clean_text"]
y_train = train_df["label"]

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_papers = vectorizer.transform(papers_df["clean_text"])

clf = MultinomialNB()
clf.fit(X_train, y_train)

# =========================
# Text cleaning
# =========================
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================
# Request model
# =========================
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# =========================
# Recommendation logic
# =========================
def recommend_papers(query: str, top_k: int = 5):
    clean_query = clean_text(query)
    query_vec = vectorizer.transform([clean_query])

    predicted_label = clf.predict(query_vec)[0]
    predicted_probs = clf.predict_proba(query_vec)[0]
    class_confidence = float(max(predicted_probs))

    sims = cosine_similarity(query_vec, X_papers).flatten()

    results = papers_df.copy()
    results["similarity"] = sims
    results["label_boost"] = (results["label"] == predicted_label).astype(int) * 0.15
    results["final_score"] = results["similarity"] + results["label_boost"]

    top_results = results.sort_values("final_score", ascending=False).head(top_k)

    recommendations = []
    for _, row in top_results.iterrows():
        summary = row["abstract"][:350] + "..." if len(row["abstract"]) > 350 else row["abstract"]
        explanation = (
            f"Recommended because it matches the predicted topic '{predicted_label}' "
            f"and is semantically similar to the query."
        )
        recommendations.append({
            "title": row["title"],
            "label": row["label"],
            "cluster": int(row["kmeans_cluster"]),
            "similarity": round(float(row["similarity"]), 4),
            "final_score": round(float(row["final_score"]), 4),
            "summary": summary,
            "explanation": explanation
        })

    return {
        "predicted_topic": predicted_label,
        "confidence": round(class_confidence, 4),
        "recommendations": recommendations
    }

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"message": "Research Paper Recommender API is running."}

@app.post("/recommend")
def recommend(request: QueryRequest):
    return recommend_papers(request.query, request.top_k)
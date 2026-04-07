import pandas as pd
import re
import nltk
import streamlit as st

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# =========================
# Page config
# =========================
st.set_page_config(page_title="Research Paper Chatbot", page_icon="🤖", layout="wide")

# =========================
# Load data
# =========================
@st.cache_data
def load_data():
    train_df = pd.read_csv("data/train_balanced.csv")
    papers_df = pd.read_csv("results/papers_with_clusters.csv")
    return train_df, papers_df

train_df, papers_df = load_data()

# =========================
# Build vectorizer + classifier
# =========================
@st.cache_resource
def build_models(train_df, papers_df):
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

    return vectorizer, clf, X_papers

vectorizer, clf, X_papers = build_models(train_df, papers_df)

# =========================
# Text cleaning
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
# Recommendation function
# =========================
def recommend_papers(query, top_k=5):
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
            f"I recommended this because it matches the predicted topic '{predicted_label}' "
            f"and is semantically similar to your request."
        )
        recommendations.append({
            "title": row["title"],
            "label": row["label"],
            "cluster": row["kmeans_cluster"],
            "similarity": round(float(row["similarity"]), 4),
            "final_score": round(float(row["final_score"]), 4),
            "summary": summary,
            "explanation": explanation
        })

    return predicted_label, round(class_confidence, 4), recommendations

# =========================
# Title
# =========================
st.title("🤖 Conversational Research Paper Chatbot")
st.caption("A research paper recommender that classifies your query, finds similar papers, and explains why they were selected.")

# =========================
# Sidebar
# =========================
st.sidebar.header("Example prompts")
example_prompts = [
    "I want papers about NLP translation models",
    "Recommend papers on recommender systems",
    "Find papers about image segmentation and object detection",
    "Show me papers on databases and query optimization",
    "Papers about deep learning and neural networks"
]
for ex in example_prompts:
    st.sidebar.write("- " + ex)

top_k = st.sidebar.slider("Number of recommendations", min_value=3, max_value=10, value=5)

# =========================
# Chat history
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I’m your research paper chatbot. 👋\n\n"
            "Tell me what topic you are interested in, and I’ll recommend relevant papers.\n\n"
            "For example, you can ask:\n"
            "- Recommend papers on recommender systems\n"
            "- Find papers about image segmentation\n"
            "- Show me papers on NLP translation"
        )
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# User input
# =========================
user_query = st.chat_input("Ask me for papers by topic, method, or research area...")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate assistant response
    predicted_label, confidence, recs = recommend_papers(user_query, top_k=top_k)

    response_text = (
        f"I understood your request as related to **{predicted_label}** "
        f"with confidence **{confidence}**.\n\n"
        f"Here are my top {top_k} recommendations:"
    )

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

        for i, rec in enumerate(recs, start=1):
            with st.expander(f"{i}. {rec['title']}"):
                st.write(f"**Predicted Label:** {rec['label']}")
                st.write(f"**Cluster:** {rec['cluster']}")
                st.write(f"**Similarity Score:** {rec['similarity']}")
                st.write(f"**Final Score:** {rec['final_score']}")
                st.write(f"**Why this paper?** {rec['explanation']}")
                st.write(f"**Summary:** {rec['summary']}")
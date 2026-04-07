import pandas as pd
import re
import nltk

from fastapi import FastAPI
from typing import Any, Dict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

app = FastAPI(title="Dialogflow Research Paper Webhook")

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


# =========================
# Recommender logic
# =========================
def recommend_papers(query, top_k=3):
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

    return predicted_label, class_confidence, top_results


def get_saved_paper_info(output_contexts, paper_number):
    title_key = f"paper{paper_number}_title"
    abstract_key = f"paper{paper_number}_abstract"

    for ctx in output_contexts:
        name = ctx.get("name", "")
        if name.endswith("/contexts/last_recommendation"):
            params = ctx.get("parameters", {})
            return params.get(title_key), params.get(abstract_key)

    return None, None
# =========================
# Root route
# =========================
@app.get("/")
def root():
    return {"message": "Dialogflow webhook is running."}


def get_context_parameter(output_contexts, context_suffix, param_name):
    for ctx in output_contexts:
        name = ctx.get("name", "")
        if name.endswith(f"/contexts/{context_suffix}"):
            return ctx.get("parameters", {}).get(param_name)
    return None
# =========================
# Dialogflow webhook route
# =========================
@app.post("/webhook")
async def dialogflow_webhook(body: Dict[str, Any]):
    query_text = body.get("queryResult", {}).get("queryText", "")
    intent_name = body.get("queryResult", {}).get("intent", {}).get("displayName", "")
    output_contexts = body.get("queryResult", {}).get("outputContexts", [])
    parameters = body.get("queryResult", {}).get("parameters", {})
    
    if not query_text:
        return {
            "fulfillmentText": "I couldn't read your query. Please try asking about a paper topic or research area."
        }
    
    if intent_name == "ask_recommended_paper_details":
        paper_number = parameters.get("paper_number", 1)

        try:
            paper_number = int(paper_number)
        except:
            paper_number = 1

        if paper_number not in [1, 2, 3]:
            return {
                "fulfillmentText": "Please ask about paper 1, paper 2, or paper 3."
            }

        title, abstract = get_saved_paper_info(output_contexts, paper_number)

        if not title:
            return {
                "fulfillmentText": "I couldn’t find that recommended paper in memory. Please ask for recommendations again first."
            }

        if "abstract" in query_text.lower():
            detail_text = abstract
            detail_type = "abstract"
        else:
            detail_text = abstract[:700] + "..." if len(abstract) > 700 else abstract
            detail_type = "summary"

        reply = (
            f"Here is the {detail_type} of paper {paper_number}:\n\n"
            f"**{title}**\n\n"
            f"{detail_text}"
        )

        return {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [reply]
                    }
                }
            ]
        }
    # If the user asks for more papers, use the last remembered topic
    if intent_name == "followup_more_papers":
        last_topic = get_context_parameter(output_contexts, "last_recommendation", "last_topic")

        if not last_topic:
            return {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": ["I don’t have a previous topic saved yet. Please first ask for papers on a topic."]
                        }
                    }
                ]
            }

        query_for_recommendation = str(last_topic)
    else:
        query_for_recommendation = query_text

    predicted_label, confidence, top_results = recommend_papers(query_for_recommendation, top_k=3)

    session = body.get("session", "")
    context_name = f"{session}/contexts/last_recommendation"

    
    reply = (
    f"Sure — based on your query, I think you're interested in **{predicted_label}** "
    f"with confidence **{confidence:.2f}**.\n\n"
    f"I found these recommended papers for you:\n"
)

    for i, (_, row) in enumerate(top_results.iterrows(), start=1):
        reply += (
            f"\n{i}. **{row['title']}** ({row['label']})"
            f"\n   This paper was recommended because it is highly related to your query."
        )

    reply += (
        "\n\nYou can now ask me things like:"
        "\n- Tell me more about paper 1"
        "\n- Summarize paper 2"
        "\n- Show me the abstract of paper 3"
    )

    return {
    "fulfillmentMessages": [
        {
            "text": {
                "text": [reply]
            }
        }
    ],
    "outputContexts": [
        {
            "name": context_name,
            "lifespanCount": 5,
            "parameters": {
                "last_topic": predicted_label,
                "last_query": query_for_recommendation,
                "paper1_title": top_results.iloc[0]["title"] if len(top_results) > 0 else "",
                "paper1_abstract": top_results.iloc[0]["abstract"] if len(top_results) > 0 else "",
                "paper2_title": top_results.iloc[1]["title"] if len(top_results) > 1 else "",
                "paper2_abstract": top_results.iloc[1]["abstract"] if len(top_results) > 1 else "",
                "paper3_title": top_results.iloc[2]["title"] if len(top_results) > 2 else "",
                "paper3_abstract": top_results.iloc[2]["abstract"] if len(top_results) > 2 else ""
            }
        }
    ]
}
"""
Fake News Classifier — Gradio Web App
Run:  python app.py
"""

import gradio as gr
import joblib
import string
import nltk
from nltk.corpus import stopwords

# ── Download NLTK data (first-run only) ──────────────────────────────────────
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

# ── Load saved pipeline ───────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"

try:
    model_data = joblib.load(MODEL_PATH)
    pipeline   = model_data["pipeline"]
    model_name = model_data.get("model_name", "Best Model")
    print(f"✅ Loaded model: {model_name}")
except FileNotFoundError:
    raise RuntimeError(
        f"Model file not found at '{MODEL_PATH}'. "
        "Please run notebooks/2_training.ipynb first to train and save the model."
    )


# ── Preprocessing (must match training exactly) ───────────────────────────────
def preprocess_text(text: str) -> str:
    """Lowercase → remove punctuation → remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ── Prediction function ───────────────────────────────────────────────────────
def predict_news(article: str):
    """
    Takes a raw news article, preprocesses it, runs inference,
    and returns the predicted label + confidence scores.
    """
    if not article.strip():
        return "⚠️ Please enter a news article.", {}

    clean = preprocess_text(article)
    label = pipeline.predict([clean])[0]
    proba = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_

    confidence = {cls: float(f"{p:.4f}") for cls, p in zip(classes, proba)}
    top_prob = max(proba)

    if label == "Fake":
        result = f"🔴 FAKE NEWS  (confidence: {top_prob*100:.1f}%)"
    else:
        result = f"🟢 REAL NEWS  (confidence: {top_prob*100:.1f}%)"

    return result, confidence


# ── Example articles ──────────────────────────────────────────────────────────
EXAMPLES = [
    [
        "Scientists at NASA confirmed today that a massive asteroid will pass "
        "safely by Earth next month, posing no threat to the planet. "
        "The announcement was made during a press conference at the Johnson Space Center."
    ],
    [
        "BREAKING: Deep state operatives have been secretly poisoning the water supply "
        "with mind-control chemicals! Patriots must wake up now — share this before "
        "they delete it!! The globalists don't want you to know the truth!!!"
    ],
    [
        "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
        "marking the tenth increase in the current tightening cycle. "
        "Fed Chair Jerome Powell said the central bank remains committed to bringing "
        "inflation back to its 2 percent target."
    ],
]

# ── Build Gradio interface ────────────────────────────────────────────────────
with gr.Blocks(
    title="Fake News Classifier",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 📰 Fake News Classifier
        **Powered by TF-IDF + Logistic Regression**

        Paste any news article below and click **Classify** to find out whether it
        is likely **Real** or **Fake** news — along with the model's confidence scores.

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            article_input = gr.Textbox(
                label="📝 News Article",
                placeholder="Paste the full text of a news article here...",
                lines=10,
            )
            classify_btn = gr.Button("🔍 Classify", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Textbox(
                label="🎯 Prediction",
                lines=2,
                interactive=False,
            )
            confidence_output = gr.Label(
                label="📊 Confidence Scores",
                num_top_classes=2,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=article_input,
        label="📚 Try These Examples",
    )

    gr.Markdown(
        f"""
        ---
        **Model used:** {model_name} &nbsp;|&nbsp;
        **Preprocessing:** Lowercase · Punctuation removal · Stopword removal · TF-IDF (10k features, bigrams)
        """
    )

    classify_btn.click(
        fn=predict_news,
        inputs=article_input,
        outputs=[prediction_output, confidence_output],
    )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(share=False)

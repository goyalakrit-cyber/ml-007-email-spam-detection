"""Interactive Streamlit website for the SMS spam detection project."""

import pandas as pd
import requests
import streamlit as st

from spamdetector.data import load_dataset, validate_frame
from spamdetector.model import (
    error_analysis,
    evaluate_model,
    preprocess_text,
    train_model,
)

st.set_page_config(
    page_title="Message Shield | Spam Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(160deg, #f8fafc 0%, #eff6ff 100%); }
    .hero { padding: 1.5rem 1.8rem; border-radius: 20px; color: white;
      background: linear-gradient(110deg, #0f172a, #1d4ed8); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2.25rem; }
    .hero p { color: #dbeafe; margin-bottom: 0; }
    </style>
    <div class="hero"><h1>🛡️ Message Shield</h1>
    <p>A transparent, interactive baseline for detecting spam in text messages.</p></div>
    """,
    unsafe_allow_html=True,
)
st.caption("The benchmark dataset contains SMS messages; it is not email-specific.")

with st.sidebar:
    st.header("Model setup")
    test_size = st.slider(
        "Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05
    )
    seed = st.number_input("Random seed", min_value=0, max_value=100_000, value=42)
    remove_stopwords = st.checkbox(
        "Remove English stopwords",
        value=False,
        help="Optional: remove common words such as “the” and “is” before TF-IDF.",
    )
    stemming = st.checkbox(
        "Apply English stemming",
        value=False,
        help="Optional: reduce words to their stems (for example, “running” to “run”).",
    )
    upload = st.file_uploader(
        "Optional labelled CSV",
        type=["csv"],
        help="Requires label and text columns with ham/spam labels.",
    )
    st.caption("Stratified splits preserve the ham/spam proportions.")

try:
    if upload is not None:
        source = validate_frame(pd.read_csv(upload))
        source_name = "Uploaded CSV"
    else:
        source = load_dataset()
        source_name = "UCI SMS Spam Collection"
    with st.spinner("Training the TF-IDF + Naive Bayes baseline..."):
        result = train_model(
            source,
            test_size=float(test_size),
            seed=int(seed),
            remove_stopwords=remove_stopwords,
            stemming=stemming,
        )
    metrics = evaluate_model(result)
except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not prepare the model: {exc}")
    st.info("Check your connection for the first dataset download, or upload a valid labelled CSV.")
    st.stop()

counts = st.columns(3)
counts[0].metric("Messages", f"{len(source):,}")
counts[1].metric("Ham", f"{(source['label'] == 'ham').sum():,}")
counts[2].metric("Spam", f"{(source['label'] == 'spam').sum():,}")
st.caption(
    f"Data: {source_name} · TF-IDF + Multinomial Naive Bayes · seed {seed} · "
    f"stopword removal {'on' if remove_stopwords else 'off'} · "
    f"stemming {'on' if stemming else 'off'}"
)

st.subheader("Held-out evaluation")
columns = st.columns(4)
for column, label, key in zip(
    columns,
    ["Accuracy", "Spam precision", "Spam recall", "Spam F1"],
    ["accuracy", "precision", "recall", "f1"],
):
    column.metric(label, f"{metrics[key]:.1%}")

matrix_col, errors_col = st.columns([1, 1.4])
with matrix_col:
    st.markdown("#### Confusion matrix")
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"],
    )
    st.dataframe(matrix, use_container_width=True)
    st.caption("Rows are actual classes; columns are model predictions.")

with errors_col:
    st.markdown("#### False positives and false negatives")
    errors = error_analysis(result)
    if errors.empty:
        st.success("No mistakes in this test split.")
    else:
        category = st.selectbox(
            "Filter errors", ["All", "False positive", "False negative"]
        )
        visible = errors if category == "All" else errors[errors["error_type"] == category]
        st.dataframe(visible, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Try a message")
st.write(
    "Text is cleaned and tokenized automatically using the selected preprocessing "
    "settings. The model updates as you type. Predictions are estimates, not guarantees; "
    "avoid entering sensitive content."
)
message = st.text_area("Message text", placeholder="Type or paste a message...", height=130)
if message.strip():
    stages = preprocess_text(
        message, remove_stopwords=remove_stopwords, stemming=stemming
    )
    with st.expander("See automatic text preprocessing"):
        st.text_area(
            "Cleaned text",
            value=stages["cleaned"],
            height=75,
            disabled=True,
        )
        st.write("**Tokens:**", ", ".join(stages["tokens"]) or "(no tokens)")
        st.caption("Token sequence sent to TF-IDF after the selected optional steps.")

    model = result["model"]
    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    classes = list(model.named_steps["classifier"].classes_)
    confidence = float(probabilities[classes.index(prediction)])
    if prediction == "spam":
        st.warning(f"Likely SPAM · model confidence {confidence:.1%}")
    else:
        st.success(f"Likely HAM · model confidence {confidence:.1%}")
    st.progress(confidence)
    st.caption("Classifier probability is not a calibrated guarantee.")

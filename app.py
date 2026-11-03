import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Message Intelligence System", page_icon="📩", layout="wide")
st.title("📩 Message Intelligence System")
st.write("A supervised-learning demo that classifies a message profile as **Legitimate (0)** or **Spam (1)**.")
st.caption("Educational prototype. Do not use this demo as the sole filter for real messages.")

DATA_PATH = "Message_Intelligence_Dataset_5200(1).xlsx"
FEATURES = [
    "message_length", "word_count", "num_urls", "num_digits",
    "num_special_chars", "spam_keyword_score", "legit_keyword_score",
    "sender_activity_score", "sender_account_age_days",
    "messages_sent_last_24h", "hour_of_day", "day_of_week"
]
TARGET = "spam_label"

@st.cache_data
def load_data():
    data = pd.read_excel(DATA_PATH)
    return data

@st.cache_resource
def train_model():
    data = load_data()
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale",
                           probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0)
    }
    return model, metrics, data

try:
    model, metrics, data = train_model()
except FileNotFoundError:
    st.error(f"Dataset file not found: {DATA_PATH}. Keep the Excel file in the same folder as app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load/train the model: {exc}")
    st.stop()

st.subheader("📊 Model test-set summary")
cols = st.columns(4)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.1%}")
st.caption("Metrics are calculated on a held-out test split. Your results may differ if the dataset changes.")

st.divider()
st.subheader("🔎 Try a message profile")
st.write("Enter the numerical features below. These are the same engineered features used by the notebook; this app does not read raw message text.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        message_length = st.number_input("Message length", min_value=0, max_value=10000, value=100)
        word_count = st.number_input("Word count", min_value=0, max_value=2000, value=18)
        num_urls = st.number_input("Number of URLs", min_value=0, max_value=100, value=0)
        num_digits = st.number_input("Number of digits", min_value=0, max_value=1000, value=0)
    with c2:
        num_special_chars = st.number_input("Special characters", min_value=0, max_value=1000, value=1)
        spam_keyword_score = st.number_input("Spam keyword score", min_value=0, max_value=100, value=0)
        legit_keyword_score = st.number_input("Legitimate keyword score", min_value=0, max_value=100, value=1)
        sender_activity_score = st.number_input("Sender activity score", min_value=0.0, max_value=1000.0, value=25.0)
    with c3:
        sender_account_age_days = st.number_input("Sender account age (days)", min_value=0.0, max_value=100000.0, value=365.0)
        messages_sent_last_24h = st.number_input("Messages sent in last 24 hours", min_value=0.0, max_value=100000.0, value=5.0)
        hour_of_day = st.slider("Hour of day", 0, 23, 12)
        day_of_week = st.slider("Day of week (Monday=0)", 0, 6, 2)
    submitted = st.form_submit_button("Classify message profile")

if submitted:
    sample = pd.DataFrame([{
        "message_length": message_length,
        "word_count": word_count,
        "num_urls": num_urls,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "spam_keyword_score": spam_keyword_score,
        "legit_keyword_score": legit_keyword_score,
        "sender_activity_score": sender_activity_score,
        "sender_account_age_days": sender_account_age_days,
        "messages_sent_last_24h": messages_sent_last_24h,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week
    }], columns=FEATURES)
    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0]
    if prediction == 1:
        st.error("🚨 Prediction: Spam message (1)")
    else:
        st.success("✅ Prediction: Legitimate message (0)")
    p1, p2 = st.columns(2)
    p1.metric("Estimated legitimate probability", f"{probabilities[0]:.1%}")
    p2.metric("Estimated spam probability", f"{probabilities[1]:.1%}")
    st.info("These are model-estimated probabilities, not a guarantee. Review borderline or important messages manually.")

with st.expander("About this dataset"):
    st.write(f"Dataset size: {len(data):,} rows × {data.shape[1]} columns.")
    st.write("Target labels: 0 = legitimate, 1 = spam.")
    st.write("The app uses engineered numeric features and excludes message ID, raw message text, and timestamp.")

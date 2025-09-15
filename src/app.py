import streamlit as st
import pandas as pd
import numpy as np
import os, joblib, time
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_xgb_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("💳 Fraud Detection System")
st.write("Upload transactions, stream them from Kaggle API, or enter details manually to detect fraud.")

# --- Feature Engineering ---
def preprocess(df):
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Hour"] = (df["Time"] // 3600) % 24
    drop_cols = [c for c in ["Class"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df

# --- Load Kaggle data ---
def load_kaggle_data():
    dataset = "mlg-ulb/creditcardfraud"
    filename = "creditcard.csv"
    save_path = "data/"
    os.makedirs(save_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    file_path = os.path.join(save_path, filename)

    # Download only if not already extracted
    if not os.path.exists(file_path):
        api.dataset_download_file(dataset, file_name=filename, path=save_path, force=True)
        
        # Kaggle saves as .zip → extract
        zip_path = os.path.join(save_path, filename + ".zip")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_path)
        os.remove(zip_path)  # clean up zip

    return pd.read_csv(file_path)

# --- Option 1: Stream Data from API ---
st.subheader("📡 Stream Transactions API")

# Initialize streaming state
if "streaming" not in st.session_state:
    st.session_state.streaming = False

def toggle_stream():
    st.session_state.streaming = not st.session_state.streaming

# Button changes text depending on state
if st.button("⏯️ Start Streaming" if not st.session_state.streaming else "⏹️ Stop Streaming", on_click=toggle_stream):
    pass

if st.session_state.streaming:
    with st.spinner("Fetching Kaggle data..."):
        data = load_kaggle_data()
        st.write("Preview of Kaggle dataset:", data.head())

        processed = preprocess(data)

        st.write("🔄 Streaming transactions in real-time...")

        placeholder = st.empty()
        fraud_count, legit_count = 0, 0
        chart = st.line_chart({"Fraud": [], "Legit": []})

        for i, row in processed.iterrows():
            if not st.session_state.streaming:
                st.warning("⏹️ Streaming stopped by user.")
                break  # stop loop immediately

            pred = model.predict(row.values.reshape(1, -1))[0]
            proba = model.predict_proba(row.values.reshape(1, -1))[0][1]

            if pred == 1:
                fraud_count += 1
                verdict = f"🚨 Fraud detected! (Prob: {proba:.2f})"
            else:
                legit_count += 1
                verdict = f"✅ Legit Transaction (Prob fraud: {proba:.2f})"

            with placeholder.container():
                st.write(f"**Transaction {i+1}:**")
                st.write(row.to_dict())
                if pred == 1:
                    st.error(verdict)
                else:
                    st.success(verdict)

            chart.add_rows({"Fraud": [fraud_count], "Legit": [legit_count]})
            time.sleep(1)  # simulate delay


# --- Option 2: Batch Predictions ---
st.subheader("📂 Upload CSV for Batch Predictions")
batch_file = st.file_uploader("Upload a CSV for batch predictions", type=["csv"], key="batch_csv")

if batch_file:
    data = pd.read_csv(batch_file)
    st.write("Preview of uploaded data:", data.head())

    processed = preprocess(data)
    preds = model.predict(processed)

    data["Fraud_Prediction"] = preds
    st.write("Predictions:", data.head())

    fraud_count = (data["Fraud_Prediction"] == 1).sum()
    st.success(f"✅ Found {fraud_count} fraudulent transactions.")

    csv_out = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv_out, "fraud_predictions.csv", "text/csv")


# --- Option 3: Manual Input with Gamification ---
st.subheader("📝 Enter Transaction Details (Gamified Mode)")

# Initialize game state
if "score" not in st.session_state:
    st.session_state.score = 0
if "level" not in st.session_state:
    st.session_state.level = 1
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "rounds" not in st.session_state:
    st.session_state.rounds = 0

with st.form("fraud_form"):
    time_val = st.number_input("Time (in seconds)", min_value=0, value=0)
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)

    # Gamification: User also guesses if it's fraud
    user_guess = st.radio("Your Guess: Is this Fraud?", ["Fraud", "Legit"])

    submitted = st.form_submit_button("Predict & Score")

    if submitted:
        input_data = pd.DataFrame([[time_val, amount, v1, v2, v3]],
                                  columns=["Time", "Amount", "V1", "V2", "V3"])

        processed = preprocess(input_data)
        prediction = model.predict(processed)[0]
        proba = model.predict_proba(processed)[0][1]

        # Show model verdict
        if prediction == 1:
            st.error(f"⚠️ Fraudulent Transaction! (Probability: {proba:.2f})")
            true_label = "Fraud"
        else:
            st.success(f"✅ Legitimate Transaction (Probability of fraud: {proba:.2f})")
            true_label = "Legit"

        # Update game state
        st.session_state.rounds += 1
        if user_guess == true_label:
            st.session_state.score += 10
            st.session_state.streak += 1
            st.success("🎉 Correct Guess! +10 points")
        else:
            st.session_state.score -= 5
            st.session_state.streak = 0
            st.error("❌ Wrong Guess! -5 points")

        # Level progression
        if st.session_state.score >= st.session_state.level * 50:
            st.session_state.level += 1
            st.balloons()
            st.success(f"🌟 Level Up! You're now Level {st.session_state.level}")

# --- Show Gamification Stats ---
st.sidebar.header("🏆 Your Game Stats")
st.sidebar.metric("Score", st.session_state.score)
st.sidebar.metric("Level", st.session_state.level)
st.sidebar.metric("Streak", st.session_state.streak)
st.sidebar.metric("Rounds Played", st.session_state.rounds)

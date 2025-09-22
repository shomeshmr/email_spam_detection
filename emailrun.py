import streamlit as st
import joblib  # <-- use joblib

# Load vectorizer + model together
vectorizer, model = joblib.load(r"D:\email_detection\spam_detector_model.pkl")

st.title("📧 Spam Detector App")

st.write("Enter a message below and check if it's **Spam** or **Not Spam**.")

# User input
user_input = st.text_area("Message", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Transform input text
        x = vectorizer.transform([user_input])
        prediction = model.predict(x)[0]

        if prediction == 1:
            st.error("🚨 This message is classified as **SPAM**.")
        else:
            st.success("✅ This message is classified as **NOT SPAM**.")

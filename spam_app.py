import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="Spam Message Classifier", page_icon="📩")

# Title and description
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's spam or not.")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_area("✏️ Type your message here")

# Prediction
if st.button("Check Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message first.")
    else:
        input_vector = tfidf.transform([user_input])
        prediction = model.predict(input_vector)[0]
        prob = model.predict_proba(input_vector)[0][1]  # probability of spam

        # Add to history
        st.session_state.history.append({
            "Message": user_input,
            "Prediction": prediction,
            "Spam Probability (%)": round(prob * 100, 2)
        })

        # Display result
        if prediction == "spam":
            st.error(f"📛 This message is **SPAM**! (Probability: {prob*100:.2f}%)")
        else:
            st.success(f"✅ This message is **NOT spam**. (Spam Probability: {prob*100:.2f}%)")

# Show history
if st.session_state.history:
    st.subheader("🕒 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    # Pie chart
    st.subheader("📊 Prediction Summary")
    counts = hist_df["Prediction"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff4b4b','#4bb543'])
    ax.axis("equal")
    st.pyplot(fig)
# Export history to CSV
if st.session_state.history:
    hist_df = pd.DataFrame(st.session_state.history)

    csv = hist_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Prediction History as CSV",
        data=csv,
        file_name='spam_prediction_history.csv',
        mime='text/csv'
    )

# Clear history button
if st.button("🧹 Clear History"):
    st.session_state.history = []
    st.experimental_rerun()

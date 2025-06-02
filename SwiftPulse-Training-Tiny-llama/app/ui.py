# app/ui.py

import streamlit as st
from chat_engine import generate_response
import pandas as pd
import os
from datetime import datetime

LOG_PATH = "data/chat_logs.csv"  # Where logs will be stored

st.set_page_config(page_title="GenAI Chatbot", layout="centered")
st.title("🤖 GenAI Chatbot (Powered by DialoGPT)")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Button to reset conversation
if st.button("🔄 Reset Chat"):
    st.session_state.history = []
    st.success("Conversation reset.")

# User input
user_input = st.text_input("You:", placeholder="Ask me something...", key="input")

# Handle user input
if user_input:
    bot_reply = generate_response(user_input)

    # Save to session history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_reply))

    # Log conversation to CSV
    timestamp = datetime.now().isoformat()
    log_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "user_input": user_input,
        "bot_response": bot_reply
    }])

    # Create folder if needed
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, index=False)
    else:
        log_entry.to_csv(LOG_PATH, mode="a", header=False, index=False)

# Display history
for sender, msg in st.session_state.history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

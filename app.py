import streamlit as st
from chatbot import create_qa_chain  # your chain logic

# Streamlit page setup
st.set_page_config(page_title="Portfolio Chatbot", layout="wide")
st.title("👨‍💼 Portfolio Chatbot Assistant")
st.markdown("Ask anything about [Your Name] — your background, skills, projects, and more!")

# Load chain
qa_chain = create_qa_chain()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("Ask a question:")

# Process query
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

# Display chat
for sender, msg in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {msg}")

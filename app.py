# app.py

import streamlit as st
from chatbot import load_documents, split_documents, create_vectorstore, create_qa_chain

st.set_page_config(page_title="Vamshi's Chatbot 💬", layout="wide")

# Initialize chatbot chain once and cache it
@st.cache_resource
def init_qa_chain():
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    qa_chain = create_qa_chain(vectorstore)
    return qa_chain

qa_chain = init_qa_chain()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and input box
st.title("💬 Vamshi's Chatbot")
user_input = st.text_input("What you wanna know about vamshi:", key="input")

# Run query on submit
if user_input:
    response = qa_chain.run(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display conversation history
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")

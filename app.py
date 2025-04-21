import streamlit as st
import time
from chatbot import load_documents, split_documents, create_vectorstore, create_qa_chain

# Page config
st.set_page_config(page_title="Vamshi's Chatbot", layout="centered")

# Title
st.title("💬 Vamshi's Chatbot")
st.markdown("Ask anything about **Vamshi**!")

# Load chain (cached)
@st.cache_resource
def load_chain():
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    return create_qa_chain(vectorstore)

qa_chain = load_chain()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores tuples of (sender, message)

# Multiline input for better UX
query = st.chat_input("Ask your question here...")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.session_state.chat_history.append(("bot", response))

# Chat display
for i, (sender, message) in enumerate(st.session_state.chat_history):
    with st.chat_message("user" if sender == "user" else "assistant"):
        # Only animate the last message if it's a bot response
        is_last = i == len(st.session_state.chat_history) - 1
        is_bot = sender == "bot"

        if is_last and is_bot:
            placeholder = st.empty()
            full_msg = ""
            for word in message.split():
                full_msg += word + " "
                placeholder.markdown(full_msg + "▌")
                time.sleep(0.05)
            placeholder.markdown(full_msg)
        else:
            st.markdown(message)

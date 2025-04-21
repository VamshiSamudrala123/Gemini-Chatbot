import streamlit as st
import time
from chatbot import load_documents, split_documents, create_vectorstore, create_qa_chain

st.title("Vamshi's Chatbot")
st.markdown("Please ask anything about Vamshi")

@st.cache_resource
def load_chain():
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    return create_qa_chain(vectorstore)

qa_chain = load_chain()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.session_state.chat_history.insert(0, ("You", query))   # newest on top
        st.session_state.chat_history.insert(0, ("Bot", response))  # newest on top

# Display chat with most recent on top
for sender, message in st.session_state.chat_history:
    if sender == "Bot":
        with st.container():
            placeholder = st.empty()
            full_message = ""
            for word in message.split():
                full_message += word + " "
                placeholder.markdown(f"**🤖 Bot:** {full_message}")
                time.sleep(0.07)  # word delay
    elif sender == "You":
        st.markdown(f"**🧑 You:** {message}")
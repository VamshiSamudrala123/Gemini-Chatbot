import streamlit as st
from chatbot import load_documents, split_documents, create_vectorstore, create_qa_chain

# UI
st.title("Vamshi's Chatbot")
st.markdown("Please ask anything about vamshi")

# Load RAG pipeline
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

# Chat input
query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

# Show chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

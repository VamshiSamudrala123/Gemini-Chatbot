import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


# Load API key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 1. Load documents from 'data/' folder
def load_documents():
    documents = []
    for filename in os.listdir("data"):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join("data", filename))
            documents.extend(loader.load())
    return documents

# 2. Split documents into chunks
def split_documents(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# 3. Create embeddings and FAISS vector store
#def create_vectorstore(chunks):
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    #print("Number of chunks:", len(chunks))
    #print("First chunk:", chunks[0] if chunks else "No chunks loaded")

    #print("Testing embeddings...")
    #print("Sample embedding:", embeddings.embed_query("Hello world"))

    #print("Testing embeddings...")
    #try:
       # result = embeddings.embed_query("Hello world")
       # print("Sample embedding length:", len(result))
    #except Exception as e:
       # print("Embedding failed:", e)


    #return FAISS.from_documents(chunks, embeddings)

# 3. Create embeddings and FAISS vector store
def create_vectorstore(chunks):
    if len(chunks) == 0:
        raise ValueError("❌ No document chunks found. Please check your 'data/' folder.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Optional: test the embedding model
    print("🔍 Testing embeddings...")
    try:
        result = embeddings.embed_query("Hello world")
        print("✅ Sample embedding length:", len(result))
    except Exception as e:
        raise RuntimeError(f"❌ Embedding failed: {e}")

    return FAISS.from_documents(chunks, embeddings)


# 4. Create RAG pipeline with Gemini
#def create_qa_chain(vectorstore):
    #retriever = vectorstore.as_retriever()
    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", temperature=0.3)
    #return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

from prompt import get_custom_prompt  # import your function

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt = get_custom_prompt()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )


# 5. Main chatbot loop
def main():
    print("📚 Loading documents...")
    documents = load_documents()
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    qa_chain = create_qa_chain(vectorstore)
    print("🤖 Chatbot is ready. Ask a question! Type 'exit' to quit.")
    
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("👋 Bye!")
            break
        response = qa_chain.run(query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()

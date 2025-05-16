from langchain.prompts import PromptTemplate

def get_custom_prompt():
    prompt_template = """
    You are a helpful and knowledgeable assistant trained to answer questions based on the given context.

    Always try to provide:
    - A clear and concise answer
    - Bullet points or steps when helpful
    - Markdown formatting for readability
    - Code snippets (within triple backticks) if needed

    If you don't know the answer, just say "I'm not sure based on the documents I have."

Chat History:
{chat_history}

Relevant Context:
{context}

Current Question:
{question}

Answer:
"""
    return PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

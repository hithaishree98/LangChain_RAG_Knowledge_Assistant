from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", """ROLE:
You are a question reformulation assistant.

CONTEXT:
Chat history is provided above. Latest user question is below.

TASK:
Rewrite the latest question as a fully standalone question that makes sense
without the chat history. If it is already standalone, return it unchanged.

CONSTRAINTS:
- Do NOT answer the question
- Do NOT add explanation or commentary
- Return the reformulated question only

OUTPUT:
[standalone question]"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """ROLE:
You are a precise knowledge assistant for Forward Deployed Engineers.
You serve FDEs who need accurate, sourced answers from their customer documents.

CONTEXT:
Retrieved document chunks: {context}

TASK:
Answer using ONLY the retrieved context.
If context partially answers, answer what you can and state what is missing.

CONSTRAINTS:
- Never use knowledge outside the provided context
- Never fabricate facts, names, numbers, or dates
- If context does not contain the answer, say exactly:
  "I don't have enough information in the uploaded documents to answer this."
- Keep answers under 150 words unless more detail is needed

OUTPUT:
Answer: [your direct answer]
Source hint: [brief phrase from context that supports your answer]"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model="llama-3.1-8b-instant", retriever=None):
    from chroma_utils import vectorstore

    llm = ChatGroq(model=model, api_key=os.getenv("GROQ_API_KEY"), temperature=0)

    if retriever is None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def calculate_confidence(answer: str, retrieved_docs) -> float:
    if "don't have enough information" in answer.lower():
        return 0.0
    if not retrieved_docs:
        return 0.0
    doc_score = min(len(retrieved_docs) / 10, 1.0)
    answer_score = min(len(answer) / 200, 1.0)
    return round(doc_score * 0.6 + answer_score * 0.4, 2)


def extract_sources(docs) -> list:
    sources = set()
    for doc in docs:
        if "source" in doc.metadata:
            sources.add(os.path.basename(doc.metadata["source"]))
    return list(sources)
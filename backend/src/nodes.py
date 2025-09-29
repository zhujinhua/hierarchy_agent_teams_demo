from vectorstore import get_or_create_vectorstore

vectorstore, retriever = get_or_create_vectorstore(
    persist_dir="data/chroma_db",
    collection_name="rag-chroma"
)


def retrieve(state):
    """Retrieve documents from vectorstore"""
    print("---RETRIEVE---")
    question = state['messages'][0].content
    # Load existing vector store
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(llm, state):
    """Generate answer using RAG chain"""
    from prompts import get_rag_chain
    print("---GENERATE---")
    question = state['question']
    documents = state["documents"]
    rag_chain = get_rag_chain(llm)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    docs_txt = format_docs(documents)
    generation = rag_chain.invoke({"context": docs_txt, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(llm, state):
    """Filter relevant documents"""
    from prompts import get_retrieval_grader

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    retrieval_grader = get_retrieval_grader(llm)
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}


def transform_query(llm, state):
    """Rewrite query for better retrieval"""
    from prompts import get_rewrite_grader

    print("---TRANSFORM QUERY---")
    question_rewriter = get_rewrite_grader(llm)
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}

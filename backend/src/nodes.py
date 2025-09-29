from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma

from prompts import rag_chain
from vectorstore import get_or_create_vectorstore

# todo global
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


def generate(state):
    """Generate answer using RAG chain"""
    print("---GENERATE---")
    question = state['question']
    documents = state["documents"]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    docs_txt = format_docs(documents)
    generation = rag_chain.invoke({"context": docs_txt, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """Filter relevant documents"""
    from prompts import retrieval_grader

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """Rewrite query for better retrieval"""
    from prompts import question_rewriter

    print("---TRANSFORM QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"documents": state["documents"], "question": better_question}


def decide_to_generate(state):
    """Decide whether to generate or rephrase query"""
    print("---ASSESS GRADED DOCUMENTS---")
    if not state["documents"]:
        print("---DECISION: TRANSFORM QUERY---")
        return "transform_query"
    print("---DECISION: GENERATE---")
    return "generate"


# todo not use
def grade_generation_v_documents_and_question(state):
    """Check if generation is grounded and answers question"""
    from prompts import hallucination_grader, answer_grader

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke({
        "documents": documents,
        "generation": generation
    })

    if hallucination_score.binary_score == "yes":
        print("---DECISION: GENERATION IS GROUNDED---")
        answer_score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        if answer_score.binary_score == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
    print("---DECISION: GENERATION NOT GROUNDED---")
    return "not supported"

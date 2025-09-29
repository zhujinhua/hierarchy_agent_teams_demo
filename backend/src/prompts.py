from dotenv import load_dotenv
from langchain import hub
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from models import GradeDocuments, GradeHallucinations, GradeAnswer

load_dotenv()

# Prompt templates
root_supervisor_prompt = (
    "You are the root supervisor. You manage high-level teams: {members}."
    "The 'adaptive_rag_team' is best for answering questions grounded in agents, prompt engineering, and adversarial attacks."
    "The 'research_team' is best for general web search and browsing tasks."
    "The 'writing_team' is best for outlining, analysis, and content creation tasks."
    "Given the current state and user request, respond with the worker to act next. Each worker will perform a task,"
    "and respond with their results. When finished, respond with FINISH. Remember return **strict JSON only**."
)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether an answer addresses / resolves a question
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", """You a question re-writer that converts an input question to a better version that is optimized
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])
rag_supervisor_prompt = (
    "You are the Adaptive RAG team supervisor. You manage these workers: {members}. "
    "Route the task through the RAG pipeline to ensure retrieval, grading, and generation. "
    "Always begin with retrieval, then proceed to grading or query rewriting, and finish with generate worker. "
    "Stop when you are confident a useful grounded answer is produced by the generate worker. "
    "When done, respond with FINISH. Remember return **strict JSON only**, eg. {{ 'next': 'FINISH' }}"
)

outline_prompt = ("You can read documents and create outlines for the document writer. "
                  "Don't ask follow-up questions.")
writer_prompt = ("You can read, write and edit documents based on note-taker's outlines. "
                 "Don't ask follow-up questions.")


# Chains
def get_retrieval_grader(llm):
    return grade_prompt | llm.with_structured_output(GradeDocuments)


def get_rewrite_grader(llm):
    return re_write_prompt | llm | StrOutputParser()


def get_rag_chain(llm):
    # RAG chain
    rag_prompt = hub.pull("rlm/rag-prompt")
    return rag_prompt | llm | StrOutputParser()

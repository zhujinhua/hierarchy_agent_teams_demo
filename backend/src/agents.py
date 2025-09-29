from typing import Literal, TypedDict, Optional, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from prompts import (rag_supervisor_prompt, root_supervisor_prompt, writer_prompt, outline_prompt)
from tools import (tavily_tool, scrape_webpages, create_outline, write_document,
                   read_document, edit_document, python_repl_tool)
from nodes import (retrieve, generate, grade_documents, transform_query)


class State(MessagesState):
    next: str
    question: Optional[str] = None
    documents: Optional[List] = None


def make_supervisor_node(
        llm: BaseChatModel,
        members: list[str],
        prompt: str | None = None,
) -> callable:
    """
    Create a supervisor node that routes tasks to workers (members) or FINISH.
    The prompt can be customized per team; otherwise, a default will be used.
    """
    options = ["FINISH"] + members

    # Default generic prompt if none is provided
    if prompt is None:
        prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members}. Given the following user request and state,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH. Remember return **strict JSON only**,"
            " e.g., {'next':'worker_name'} or {'next':'FINISH'}."
        )

    class Router(TypedDict):
        """Worker to route to next. If no workers needed, route to FINISH."""
        next: Literal[*options]

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router node for managing workers."""
        state = state or {"messages": []}
        messages = [{"role": "system", "content": prompt}] + [state["messages"][-1]]
        # messages = [{"role": "system", "content": prompt}] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages) \
                    or llm.with_structured_output(Router).invoke([messages[-1]])
        goto = response["next"] if response else "FINISH"
        print(f"[Supervisor] Decided to go to: {goto}")
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def create_research_team(llm: BaseChatModel):
    search_agent = create_react_agent(llm, tools=[tavily_tool])
    web_scraper_agent = create_react_agent(llm, tools=[scrape_webpages])

    def search_node(state: State) -> Command[Literal["supervisor"]]:
        result = search_agent.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="search")]},
                       goto="supervisor")

    def web_scraper_node(state: State) -> Command[Literal["supervisor"]]:
        result = web_scraper_agent.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="web_scraper")]},
                       goto="supervisor")

    research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])
    builder = StateGraph(State)
    builder.add_node("supervisor", research_supervisor_node)
    builder.add_node("search", search_node)
    builder.add_node("web_scraper", web_scraper_node)
    builder.add_edge(START, "supervisor")
    return builder.compile()


def create_doc_team(llm: BaseChatModel):
    outline_agent = create_react_agent(llm, tools=[create_outline, write_document], prompt=outline_prompt)
    doc_writer_agent = create_react_agent(llm, tools=[write_document, edit_document, read_document],
                                          prompt=writer_prompt)
    chart_generating_agent = create_react_agent(llm, tools=[read_document, python_repl_tool])

    def outline_node(state: State) -> Command[Literal["supervisor"]]:
        result = outline_agent.invoke(state)
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="outline_taker")]},
            goto="supervisor")

    def doc_writing_node(state: State) -> Command[Literal["supervisor"]]:
        result = doc_writer_agent.invoke(state)
        return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="doc_writer")]},
                       goto="supervisor")

    def chart_generating_node(state: State) -> Command[Literal["supervisor"]]:
        result = chart_generating_agent.invoke(state)
        return Command(
            update={"messages": [HumanMessage(content=result["messages"][-1].content, name="chart_generator")]},
            goto="supervisor")

    doc_supervisor = make_supervisor_node(llm, ["doc_writer", "outline_taker", "chart_generator"])

    builder = StateGraph(State)
    builder.add_node("supervisor", doc_supervisor)
    builder.add_node("doc_writer", doc_writing_node)
    builder.add_node("outline_taker", outline_node)
    builder.add_node("chart_generator", chart_generating_node)
    builder.add_edge(START, "supervisor")
    return builder.compile()


def create_adaptive_rag_team(llm: BaseChatModel):
    def retrieve_node(state: State) -> Command[Literal["supervisor"]]:
        result = retrieve(state)
        docs_txt = "\n\n".join([d.page_content for d in result["documents"]])
        return Command(
            update={"messages": [HumanMessage(content=f"Retrieved {len(result['documents'])} docs:\n{docs_txt}",
                                              name="retrieve")],
                    "documents": result["documents"], "question": result["question"]},
            goto="supervisor"
        )

    def grade_documents_node(state: State) -> Command[Literal["supervisor"]]:
        result = grade_documents(llm, state)
        docs_txt = "\n\n".join([d.page_content for d in result["documents"]])
        return Command(
            update={"messages": [HumanMessage(content=f"Relevant docs after grading:\n{docs_txt}",
                                              name="grade_documents")],
                    "documents": result["documents"], "question": result["question"]},
            goto="supervisor"
        )

    def transform_query_node(state: State) -> Command[Literal["supervisor"]]:
        result = transform_query(llm, state)
        return Command(
            update={"messages": [HumanMessage(content="Transformed query", name="transform_query")],
                    "documents": result["documents"], "question": result["question"]},
            goto="supervisor"
        )

    def generate_node(state: State) -> Command[Literal["supervisor"]]:
        result = generate(llm, state)
        return Command(
            update={"messages": [HumanMessage(content=result["generation"], name="generate")]},
            goto="supervisor"
        )

    rag_members = ["retrieve", "grade_documents", "transform_query", "generate"]
    formatted_prompt = rag_supervisor_prompt.format(members=", ".join(rag_members))
    adaptive_supervisor = make_supervisor_node(
        llm, rag_members, formatted_prompt
    )

    builder = StateGraph(State)
    builder.add_node("supervisor", adaptive_supervisor)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("transform_query", transform_query_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "supervisor")
    return builder.compile()


def create_hierarchical_graphs(llm: BaseChatModel):
    def call_research_team(state: State) -> Command[Literal["supervisor"]]:
        research_graph = create_research_team(llm)
        response = research_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={"messages": [HumanMessage(content=response["messages"][-1].content, name="research_team")]},
            goto="supervisor")

    def call_writing_team(state: State) -> Command[Literal["supervisor"]]:
        doc_writing_graph = create_doc_team(llm)
        response = doc_writing_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={"messages": [HumanMessage(content=response["messages"][-1].content, name="writing_team")]},
            goto="supervisor")

    def call_adaptive_rag_team(state: State) -> Command[Literal["supervisor"]]:
        rag_graph = create_adaptive_rag_team(llm)
        response = rag_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={"messages": [HumanMessage(content=response["messages"][-1].content, name="adaptive_rag_team")]},
            goto="supervisor")

    members = ["research_team", "writing_team", "adaptive_rag_team"]
    formatted_root_prompt = root_supervisor_prompt.format(members=", ".join(members))
    teams_supervisor_node = make_supervisor_node(llm, members, formatted_root_prompt)

    super_builder = StateGraph(State)
    super_builder.add_node("supervisor", teams_supervisor_node)
    super_builder.add_node("research_team", call_research_team)
    super_builder.add_node("writing_team", call_writing_team)
    super_builder.add_node("adaptive_rag_team", call_adaptive_rag_team)

    super_builder.add_edge(START, "supervisor")
    super_builder.add_edge("supervisor", END)

    return super_builder.compile()

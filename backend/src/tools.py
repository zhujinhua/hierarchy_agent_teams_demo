from typing import List, Optional, Annotated, Dict
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_experimental.utilities import PythonREPL

# Load environment variables
load_dotenv()

# Initialize temporary working directory
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

# Initialize search tool
tavily_tool = TavilySearch(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    Scrapes content from multiple web pages using their URLs.

    Args:
        urls: List of web page URLs to scrape content from

    Returns:
        Formatted string containing scraped content from all URLs,
        with each document wrapped in <Document> tags including its title
    """
    if not urls:
        return "Error: No URLs provided for scraping"

    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()

        return "\n\n".join([
            f'<Document name="{doc.metadata.get("title", "Untitled")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ])
    except Exception as e:
        return f"Error scraping webpages: {str(e)}"


@tool
def create_outline(
        points: Annotated[List[str], "List of main points or sections for the outline"],
        file_name: Annotated[str, "File path to save the outline (within working directory)"]
) -> str:
    """
    Creates and saves an outline file from a list of points.

    Args:
        points: List of main points to include in the outline
        file_name: Name/path of the file to save the outline

    Returns:
        Success message with file name, or error message if failed
    """
    if not points:
        return "Error: No points provided for outline creation"

    if not file_name:
        return "Error: No file name provided"

    try:
        file_path = WORKING_DIRECTORY / file_name
        with file_path.open("w") as file:
            for i, point in enumerate(points, 1):
                file.write(f"{i}. {point}\n")
        return f"Outline successfully saved to {file_name}"
    except Exception as e:
        return f"Error creating outline: {str(e)}"


@tool
def read_document(
        file_name: Annotated[str, "File path to read from (within working directory)"],
        start: Annotated[Optional[int], "Start line index (0-based, default: 0)"] = None,
        end: Annotated[Optional[int], "End line index (exclusive, default: end of file)"] = None
) -> str:
    """
    Reads and returns content from a specified document file.

    Args:
        file_name: Name/path of the file to read
        start: Optional start line index (0-based)
        end: Optional end line index (exclusive)

    Returns:
        Content of the file (or specified range) as a string,
        or error message if file not found or read fails
    """
    if not file_name:
        return "Error: No file name provided"

    file_path = WORKING_DIRECTORY / file_name
    if not file_path.exists():
        return f"Error: File '{file_name}' not found"

    try:
        with file_path.open("r") as file:
            lines = file.readlines()

        start_idx = start if start is not None else 0
        # Ensure start index is within valid range
        start_idx = max(0, min(start_idx, len(lines)))

        return "\n".join(lines[start_idx:end])
    except Exception as e:
        return f"Error reading document: {str(e)}"


@tool
def write_document(
        content: Annotated[str, "Text content to write into the document"],
        file_name: Annotated[str, "File path to save the document (within working directory)"]
) -> str:
    """
    Writes content to a specified document file.

    Args:
        content: Text content to be written
        file_name: Name/path of the file to save

    Returns:
        Success message with file name, or error message if failed
    """
    if not file_name:
        return "Error: No file name provided"

    try:
        file_path = WORKING_DIRECTORY / file_name
        with file_path.open("w") as file:
            file.write(content)
        return f"Document successfully saved to {file_name}"
    except Exception as e:
        return f"Error writing document: {str(e)}"


@tool
def edit_document(
        file_name: Annotated[str, "Path of the document to be edited."],
        inserts: Annotated[
            Dict[int, str],
            "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
        ],
) -> Annotated[str, "Path of the edited document file."]:
    """
        Edit a document by inserting text at specific line numbers.

        Args:
            file_name: Name/path of the file to be edited
            inserts: the lines to insert at specific line numbers
        Returns:
            path of the edited document
        """

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Initialize Python REPL tool
python_repl = PythonREPL()


@tool
def python_repl_tool(
        code: Annotated[str, "Python code to execute (for chart generation or data processing)"]
) -> str:
    """
    Executes Python code in a REPL environment. Useful for data processing,
    calculations, or generating charts.

    Args:
        code: Valid Python code to execute

    Returns:
        Execution result or error message if execution fails
    """
    if not code.strip():
        return "Error: No Python code provided"

    try:
        result = python_repl.run(code)
        return f"Successfully executed:\nStdout: {result}"
    except Exception as e:
        return f"Execution failed: {repr(e)}"

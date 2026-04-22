from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import Any
from rag.pdf import PDFRetrievalChain


load_dotenv(override=True)


def create_retriever() -> Any:
    """"""
    # PDF 문서를 로드합니다.
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "data", "SPRI_AI_Brief_2025_08.pdf")
    pdf = PDFRetrievalChain([pdf_path]).create_chain()

    # retriever와 chain을 생성합니다.
    pdf_retriever = pdf.retriever

    return pdf_retriever


# Initialize FastMCP server with configuration
# 노트북의 streamable_http 클라이언트(http://127.0.0.1:8002/mcp)와 맞춥니다.
mcp = FastMCP(
    "Retriever",
    instructions=(
        "You expose get_current_time for the user's local time questions, "
        "and retrieve for searching the PDF knowledge base."
    ),
    host="127.0.0.1",
    port=8002,
    streamable_http_path="/mcp",
)


@mcp.tool()
async def get_current_time(timezone_name: str = "Asia/Seoul") -> str:
    """Return the current date and time in the given IANA timezone (default: Asia/Seoul)."""
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo("Asia/Seoul")
    now = datetime.now(tz)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")


@mcp.tool()
async def retrieve(query: str) -> str:
    """
    Retrieves information from the document database based on the query.

    This function creates a retriever, queries it with the provided input,
    and returns the concatenated content of all retrieved documents.

    Args:
        query (str): The search query to find relevant information

    Returns:
        str: Concatenated text content from all retrieved documents
    """
    retriever = create_retriever()

    # Use the invoke() method to get relevant documents based on the query
    retrieved_docs = retriever.invoke(query)

    # Join all document contents with newlines and return as a single string
    return "\n".join([doc.page_content for doc in retrieved_docs])


if __name__ == "__main__":
    # Streamable HTTP (uvicorn) — MultiServerMCPClient의 transport="streamable_http"와 대응
    mcp.run(transport="streamable-http")

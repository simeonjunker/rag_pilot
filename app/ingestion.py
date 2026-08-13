import requests
import trafilatura
from urllib.parse import urlparse, unquote
from app.config import USER_AGENT
from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


def load_wikipedia_url(url: str, user_agent: tuple) -> dict:
    """
    Load a Wikipedia article from the given URL and return it as a Document.

    Args:
        url (str): The URL of the Wikipedia article.
        user_agent (tuple): A tuple containing the user agent string to use for the request.

    Raises:
        ValueError: If the text could not be extracted from the URL.

    Returns:
        dict: The extracted Wikipedia article as a dictionary containing the page content and metadata.
    """

    path = urlparse(url).path

    title = unquote(path.removeprefix("/wiki/"))

    api_url = "https://en.wikipedia.org/api/rest_v1/page/html/" + title

    response = requests.get(
        api_url,
        headers={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip",
        },
        timeout=30,
    )
    response.raise_for_status()

    text = trafilatura.extract(
        response.text,
        include_links=False,
        include_images=False,
        include_tables=False,
    )

    if not text:
        raise ValueError(f"Could not extract text from {url}")

    return {
        "page_content": text,
        "metadata": {
            "source": url,
            "title": title.replace("_", " "),
        },
    }


def fetch_docs(urls: list[str]) -> list[dict]:
    """
    Download and parse Wikipedia pages, given a list of URLs.

    Args:
        urls (list[str]): A list of Wikipedia article URLs.

    Returns:
        list[dict]: A list of dictionaries containing the page content and metadata for each Wikipedia article.
    """
    docs = []
    for url in urls:
        doc = load_wikipedia_url(url, user_agent=USER_AGENT)
        docs.append(doc)
    return docs


def chunk_docs(
    docs: list[dict], chunk_size: int = 300, chunk_overlap: int = 50
) -> list[str]:
    """
    Create text chunks from a list of input documents.

    Args:
        docs (list[dict]): A list of dictionaries containing the page content and metadata.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 300.
        chunk_overlap (int, optional): The number of overlapping characters between chunks. Defaults to 50.

    Returns:
        list[str]: A list of text chunks extracted from the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = []
    for doc in docs:
        content = doc.get("page_content")
        metadata = doc.get("metadata", {})
        
        document_chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(document_chunks):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                    id=f"{metadata.get('source', 'unknown')}_chunk_{i}",
                )
            )

    return chunks


def save_to_json(docs: list[dict], save_path: Path) -> None:
    """
    Save the downloaded Wikipedia pages to disk.

    Args:
        docs (list[dict]): A list of JSON serializable documents (Wikipedia articles).
        save_path (Path): The path where the JSON file should be saved.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)

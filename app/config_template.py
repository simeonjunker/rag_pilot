from pathlib import Path

# ---------------------------------------------------------------------------
# Project root and data directories
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

# ---------------------------------------------------------------------------
# Constants for Wikipedia ingestion and querying
# ---------------------------------------------------------------------------

WIKI_URLS = [
    "https://en.wikipedia.org/wiki/Saxony",
    "https://en.wikipedia.org/wiki/Dresden",
    "https://en.wikipedia.org/wiki/Leipzig",
    "https://en.wikipedia.org/wiki/Bautzen",
    "https://en.wikipedia.org/wiki/Chemnitz",
    "https://en.wikipedia.org/wiki/Zwickau",
    "https://en.wikipedia.org/wiki/Plauen",
    "https://en.wikipedia.org/wiki/G%C3%B6rlitz",
    "https://en.wikipedia.org/wiki/Freiberg",
    "https://en.wikipedia.org/wiki/Freital",
    "https://en.wikipedia.org/wiki/Pirna",
    "https://en.wikipedia.org/wiki/Saxon_cuisine",
    "https://en.wikipedia.org/wiki/Sorbian_languages",
]

USER_AGENT = (
    "MyRAGProject/0.1 "
    "(https://github.com/yourusername/my-rag-project; "
    "your-email@example.com)"
)

# ---------------------------------------------------------------------------
# Constants for chunking and retrieval
# ---------------------------------------------------------------------------

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

RETRIEVAL_K = 10

# ---------------------------------------------------------------------------
# Constants for LLM configuration
# ---------------------------------------------------------------------------

LLM_BASE_URL = "URL_TO_YOUR_vLLM_SERVER"  # Replace with your actual vLLM server URL
LLM_TEMP = 0.0

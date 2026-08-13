import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project root and data directories, load .env file
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

load_dotenv(PROJECT_ROOT / ".env")

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
    f"({os.getenv('USER_AGENT_REPO')}; "
    f"{os.getenv('USER_AGENT_MAIL')})"
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

LLM_TEMP = 0.0
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
    "https://en.wikipedia.org/wiki/Albertinum",
    "https://en.wikipedia.org/wiki/Annaberg-Buchholz",
    "https://en.wikipedia.org/wiki/Aue-Bad_Schlema",
    "https://en.wikipedia.org/wiki/Bachfest_Leipzig",
    "https://en.wikipedia.org/wiki/Bautzen",
    "https://en.wikipedia.org/wiki/Bischofswerda",
    "https://en.wikipedia.org/wiki/Borna",
    "https://en.wikipedia.org/wiki/Chemnitz",
    "https://en.wikipedia.org/wiki/Crimmitschau",
    "https://en.wikipedia.org/wiki/Culture_in_Dresden",
    "https://en.wikipedia.org/wiki/D%C3%B6beln",
    "https://en.wikipedia.org/wiki/Delitzsch",
    "https://en.wikipedia.org/wiki/Dippoldiswalde",
    "https://en.wikipedia.org/wiki/Dresden",
    "https://en.wikipedia.org/wiki/Dresden_Academy_of_Fine_Arts",
    "https://en.wikipedia.org/wiki/Dresden_Castle",
    "https://en.wikipedia.org/wiki/Filmfest_Dresden",
    "https://en.wikipedia.org/wiki/Dresden_Music_Festival",
    "https://en.wikipedia.org/wiki/Dresden_State_Art_Collections",
    "https://en.wikipedia.org/wiki/Stollen",
    "https://en.wikipedia.org/wiki/Eilenburg",
    "https://en.wikipedia.org/wiki/Elbe",
    "https://en.wikipedia.org/wiki/Elbe_Sandstone_Mountains",
    "https://en.wikipedia.org/wiki/Erzgebirge",
    "https://en.wikipedia.org/wiki/Freiberg",
    "https://en.wikipedia.org/wiki/Freital",
    "https://en.wikipedia.org/wiki/G%C3%B6rlitz",
    "https://en.wikipedia.org/wiki/Gemäldegalerie_Alte_Meister",
    "https://en.wikipedia.org/wiki/Gewandhaus",
    "https://en.wikipedia.org/wiki/Green_Vault",
    "https://en.wikipedia.org/wiki/Grimma",
    "https://en.wikipedia.org/wiki/Hoyerswerda",
    "https://en.wikipedia.org/wiki/Kamenz",
    "https://en.wikipedia.org/wiki/L%C3%B6bau",
    "https://en.wikipedia.org/wiki/Leipzig",
    "https://en.wikipedia.org/wiki/Bach_Archive",
    "https://en.wikipedia.org/wiki/Leipzig_Bay",
    "https://en.wikipedia.org/wiki/Leipzig_Book_Fair",
    "https://en.wikipedia.org/wiki/Leipzig_Opera",
    "https://en.wikipedia.org/wiki/Leipzig_University",
    "https://en.wikipedia.org/wiki/Leipziger_Allerlei",
    "https://en.wikipedia.org/wiki/Lower_Sorbian",
    "https://en.wikipedia.org/wiki/Lusatia",
    "https://en.wikipedia.org/wiki/Marienberg",
    "https://en.wikipedia.org/wiki/Markkleeberg",
    "https://en.wikipedia.org/wiki/Meissen",
    "https://en.wikipedia.org/wiki/Meissen_porcelain",
    "https://en.wikipedia.org/wiki/Mittweida",
    "https://en.wikipedia.org/wiki/New_Leipzig_School",
    "https://en.wikipedia.org/wiki/Oelsnitz",
    "https://en.wikipedia.org/wiki/Ore_Mountains",
    "https://en.wikipedia.org/wiki/Pflaumentoffel",
    "https://en.wikipedia.org/wiki/Pirna",
    "https://en.wikipedia.org/wiki/Plauen",
    "https://en.wikipedia.org/wiki/Radeberg",
    "https://en.wikipedia.org/wiki/Radebeul",
    "https://en.wikipedia.org/wiki/Reichenbach_im_Vogtland",
    "https://en.wikipedia.org/wiki/Riesa",
    "https://en.wikipedia.org/wiki/Saxon_Switzerland",
    "https://en.wikipedia.org/wiki/Saxon_cuisine",
    "https://en.wikipedia.org/wiki/Saxony",
    "https://en.wikipedia.org/wiki/Semperoper",
    "https://en.wikipedia.org/wiki/Sorbian_languages",
    "https://en.wikipedia.org/wiki/Sorbian_people",
    "https://en.wikipedia.org/wiki/Sorbs",
    "https://en.wikipedia.org/wiki/St._Nicholas_Church,_Leipzig",
    "https://en.wikipedia.org/wiki/St._Thomas_Church,_Leipzig",
    "https://en.wikipedia.org/wiki/Striezelmarkt",
    "https://en.wikipedia.org/wiki/Thomanerchor",
    "https://en.wikipedia.org/wiki/Torgau",
    "https://en.wikipedia.org/wiki/Upper_Lusatia",
    "https://en.wikipedia.org/wiki/Upper_Saxon_German",
    "https://en.wikipedia.org/wiki/Upper_Sorbian",
    "https://en.wikipedia.org/wiki/Vogtland",
    "https://en.wikipedia.org/wiki/Weißwasser",
    "https://en.wikipedia.org/wiki/Werdau",
    "https://en.wikipedia.org/wiki/Wurzen",
    "https://en.wikipedia.org/wiki/Zittau",
    "https://en.wikipedia.org/wiki/Zwickau",
    "https://en.wikipedia.org/wiki/Zwinger",
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
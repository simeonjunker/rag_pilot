import json
import argparse
from langchain_chroma import Chroma
from app.ingestion import fetch_docs, chunk_docs, save_to_json
from app.retrieval import embeddings
from app.config import RAW_DATA_DIR, CHROMA_DIR, WIKI_URLS, CHUNK_SIZE, CHUNK_OVERLAP


def main(args):
    """
    Get raw documents from Wikipedia (or local file),
    chunk the documents and create Chroma vector database.

    Args:
        args (_type_): _description_
    """

    # get raw docs from Wikipedia or load from file
    if args.force_download or not (RAW_DATA_DIR / "raw_docs.json").exists():
        print("Fetching Wikipedia pages...")
        docs = fetch_docs(WIKI_URLS)
        out_path = RAW_DATA_DIR / "raw_docs.json"
        save_to_json(docs, out_path)
        print(f"Saved {len(docs)} pages to {out_path}")
    else:
        print("Loading existing Wikipedia pages...")
        with open(RAW_DATA_DIR / "raw_docs.json", "r") as f:
            docs = json.load(f)

    # chunk the docs
    chunks = chunk_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks from {len(docs)} pages.")
  
    # make vector database from chunks and save to disk
    vector_db = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="wikipedia",
        persist_directory=CHROMA_DIR,
    )
    print(f"Saved vector database to {CHROMA_DIR}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia pages and save them as JSON."
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of Wikipedia pages even if they already exist.",
    )

    args = parser.parse_args()

    main(args)

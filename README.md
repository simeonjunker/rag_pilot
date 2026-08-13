# RAG Pilot

A small retrieval-augmented chatbot based on LangChain. 
It indexes selected Wikipedia pages into a Chroma vector store, retrieves relevant passages, and uses a chat model plus simple tools for Wikipedia search and weather lookup.

## Features

- Downloads and preprocesses Wikipedia
- Splits documents into chunks and stores them in Chroma
- Uses a chat agent with retrieval and weather tools
- Connects to a vLLM-compatible OpenAI API endpoint

## Setup

Install the project:

```bash
pip install -e .
```

Create `.env` file and fill in the necessary info:
```bash
cp .env.template .env
nano .env
```

## Build the vector store

Run the ingestion script to fetch the source pages and create the local Chroma database in data/chroma:

```bash
python scripts/ingest.py --force_download
```

You can omit --force_download to reuse the cached raw documents in data/raw/raw_docs.json.

## Use the agent

Example:

```python
from app.agent import ChatAgent

agent = ChatAgent()
agent.query("What are good places to visit in Dresden?")
```

## Project structure

- app/: core agent, ingestion, retrieval, tool, and LLM setup code
- scripts/: helper scripts, including vector store ingestion
- data/: cached raw documents and persisted Chroma data
- tests/: pytest-based tests
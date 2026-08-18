# RAG Pilot

A small retrieval-augmented chatbot based on LangChain. 
It indexes selected Wikipedia pages into a Chroma vector store, retrieves relevant passages, and uses a chat model plus simple tools for Wikipedia search and weather lookup.

## Features

- Downloads and preprocesses Wikipedia
- Splits documents into chunks and stores them in Chroma
- Connects to a vLLM-compatible OpenAI API endpoint
- Provides the LLM-based agent with retrieval and weather tools
- Provides access to the chatbot via FastAPI and a browser UI

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

## Start the server

Start the server with the following command:

```bash
uvicorn app.api:app --reload
```

After this, go to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the chat UI.

## Use the agent without a server

Example:

```python
from app.agent import ChatAgent

agent = ChatAgent()

# verbose outputs with tool calls
agent.verbose_query("Are there nice parks in Leipzig?")

# stream responses, no tool calls
await agent.stream_query("How about outside the city?")
```

## Project structure

- app/: core agent, ingestion, retrieval, tool, LLM setup and API code
- scripts/: helper scripts (currently only vector store ingestion)
- data/: cached raw documents and persisted Chroma data
- tests/: pytest-based tests
- static/: HTML/CSS/JS code for the browser UI
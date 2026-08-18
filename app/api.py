from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.agent import ChatAgent

agent = ChatAgent()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/chat")
async def chat(question, thread_id="1"):
    return StreamingResponse(
        agent.stream_answer(question, thread_id=thread_id),
        media_type="text/plain",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

from langchain_community.chat_models import ChatDeepInfra
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import HumanMessage, SystemMessage

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

import httpx

import uvicorn



async def handle_ollama_stream(websocket, query: str, model: str = "granite3.3:2b"):
    async with httpx.AsyncClient() as client:
        url = "http://localhost:11434/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "stream": True
        }

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        await websocket.send_text(line)
        except Exception as e:
            await websocket.send_text(f"[ERROR] Ollama stream failed: {e}")
        finally:
            await websocket.send_text("[END]")


class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.websocket.send_text(token)

async def handle_langchain_stream(websocket, query: str, context: str, lang: str = "English", size: int = 100, model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    callback = WebSocketCallbackHandler(websocket)

    llm = ChatDeepInfra(
        model=model,
        temperature=0.7,
        streaming=True,
        callbacks=[callback]
    )

    messages = [
        SystemMessage(
            content=(
                f"You are an expert assistant designed to provide accurate responses.\n"
                f"Context in {lang}:\n{context}"
            )
        ),
        HumanMessage(
            content=(
                f"The {lang} question is:\n{query}\n"
                f"Respond in {lang} language. Limit your response to {size} {lang} words. "
                f"Offer assistance for follow-up question."
            )
        )
    ]

    try:
        await llm.ainvoke(messages)
    except Exception as e:
        await websocket.send_text(f"[ERROR] LangChain stream failed: {e}")
    finally:
        await websocket.send_text("[END]")



app = FastAPI()

# Optional: Enable CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()

    mode = data.get("mode", "ollama")  # "ollama" or "langchain"
    query = data["query"]
    context = data.get("context", "")
    lang = data.get("lang", "English")
    size = data.get("size", 100)
    model = data.get("model")  # optional override

    if mode == "langchain":
        await handle_langchain_stream(
            websocket,
            query=query,
            context=context,
            lang=lang,
            size=size,
            model=model or "meta-llama/Meta-Llama-3-8B-Instruct"
        )
    else:
        await handle_ollama_stream(
            websocket,
            query=query,
            model=model or "granite3.3:2b"
        )




def main():
    uvicorn.run("deepInfraRag:app", host="0.0.0.0", port=5990, reload=True)


if __name__ == "__main__":
    main()


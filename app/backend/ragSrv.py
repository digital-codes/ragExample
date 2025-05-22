import json
import asyncio
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager


# Read tokens.json at startup
with open("tokens.json", "r") as f:
    USERS = {u['username']: u['token'] for u in json.load(f)['users']}
    TOKENS = set(USERS.values())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Server is starting...")

    try:
        yield
    finally:
        # Shutdown logic (runs on Ctrl-C / SIGINT, SIGTERM, etc.)
        print("Server is shutting down gracefully!")
        # Place any cleanup code here (e.g., close files, db, etc.)

app = FastAPI(lifespan=lifespan)


@app.post("/api/token")
async def get_token(request: Request):
    data = await request.json()
    username = data.get("username")
    token = USERS.get(username)
    if token:
        return {"token": token}
    else:
        return JSONResponse({"error": "Invalid username"}, status_code=401)

@app.post("/api/setup")
async def setup_config(request: Request):
    data = await request.json()
    # handle config here if needed
    return {"status": "ok", "config": data}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get('token')
    if not token or token not in TOKENS:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "query":
                query = msg["data"]
                try:
                    async with httpx.AsyncClient() as client:
                        url = "http://localhost:11434/api/chat"
                        headers = {"Content-Type": "application/json"}
                        payload = {
                            "model": "llama3",  # Replace with your actual Ollama model
                            "messages": [{"role": "user", "content": query}],
                            "stream": True
                        }
                        try:
                            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                                async for chunk in resp.aiter_lines():
                                    if chunk.strip():
                                        try:
                                            delta = json.loads(chunk)
                                            content = delta.get("message", {}).get("content")
                                            if content:
                                                await websocket.send_text(json.dumps({"type": "llm_stream", "data": content}))
                                        except Exception:
                                            continue
                            await websocket.send_text(json.dumps({"type": "llm_end"}))
                        except httpx.ConnectError:
                            # Send error message to frontend
                            print("LLM backend not available")
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "data": "LLM backend is not available. Please start Ollama."
                            }))
                            await websocket.send_text(json.dumps({"type": "llm_end"}))
                except Exception as e:
                    # For any other error, report to frontend
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": f"Unexpected error: {str(e)}"
                    }))
                    await websocket.send_text(json.dumps({"type": "llm_end"}))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("Server stopped with Ctrl-C.")
    

# 3. Frontend Tweaks

# When requesting a token, POST your username to /api/token, then use that token in the WebSocket query string.
# (If you want real auth, you’d do password + JWT, but for dev/demo, this is the way!)
# 4. Summary

#     tokens.json contains multiple users/tokens for dev/multi-user realism.

#     FastAPI loads tokens on startup, checks on each websocket connect.

#     Ollama LLM streaming is handled live, and messages are streamed to Vue3 frontend.

#     You can swap "model": "llama3" for your favorite Ollama model.
    
    
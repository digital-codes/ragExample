import asyncio
import websockets
import json

async def test_ws():
    token = "alice123"  # <-- replace with a valid token
    uri = f"ws://localhost:5990/ws?token={token}"
    async with websockets.connect(uri) as websocket:
        # Send a test query
        query = "What is the capital of France?"
        await websocket.send(json.dumps({
            "action": "query",
            "data": query
        }))

        # Receive streamed response
        print("LLM response:")
        async for message in websocket:
            msg = json.loads(message)
            if msg["type"] == "llm_stream":
                print(msg["data"], end="", flush=True)
            elif msg["type"] == "llm_end":
                print("\n[Stream end]")
                break
            elif msg["type"] == "error":
                print(f"\n{msg['data']}")
                break

if __name__ == "__main__":
    asyncio.run(test_ws())

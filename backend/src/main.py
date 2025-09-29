import json

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.callbacks import BaseCallbackHandler

from agents import create_hierarchical_graphs

load_dotenv()
app = FastAPI(title="Hierarchical Agent Teams Demo")


def serialize_state(state):
    if isinstance(state, dict):
        return {k: [str(m) for m in v] if isinstance(v, list) else str(v) for k, v in state.items()}
    return str(state)


class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs):
        try:
            await self.websocket.send_text(json.dumps({"event": "token", "data": token}))
        except Exception:
            pass

    async def on_llm_end(self, response, **kwargs):
        await self.websocket.send_text(json.dumps({"event": "done"}))


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive prompt from client
        data = await websocket.receive_text()
        msg = json.loads(data)
        prompt = msg.get("prompt", "")

        handler = WebSocketCallbackHandler(websocket)
        llm = ChatTongyi(model="qwen-plus", streaming=True, callbacks=[handler])
        # Build hierarchical graph with supervisor
        graph = create_hierarchical_graphs(llm)

        await websocket.send_text(json.dumps({"event": "log", "data": "Supervisor starting"}))

        # final_state = None
        async for state in graph.astream(
                {"messages": [("user", prompt)]},
                {"recursion_limit": 50}
        ):
            # final_state = state
            await websocket.send_text(
                json.dumps({"event": "state", "data": serialize_state(state)})
            )

        await websocket.send_text(json.dumps({"event": "log", "data": "Supervisor finished"}))
        await websocket.send_text(json.dumps({"event": "final", "data": "All teams finished"}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(json.dumps({"event": "error", "data": str(e)}))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)

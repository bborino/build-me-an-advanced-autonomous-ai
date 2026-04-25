from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from autonomous_assistant.assistant import AutonomousAssistant


class RunRequest(BaseModel):
    goal: str
    session_id: str | None = None


def create_app() -> FastAPI:
    assistant = AutonomousAssistant()
    app = FastAPI(title="Autonomous Assistant", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities() -> dict:
        return assistant.capabilities()

    @app.post("/run")
    def run(request: RunRequest) -> dict:
        report = assistant.run(goal=request.goal, session_id=request.session_id)
        return report.model_dump()

    return app


app = create_app()


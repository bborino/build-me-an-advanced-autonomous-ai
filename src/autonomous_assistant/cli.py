from __future__ import annotations

import json

import typer
import uvicorn

from autonomous_assistant.api import create_app
from autonomous_assistant.assistant import AutonomousAssistant


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def capabilities() -> None:
    assistant = AutonomousAssistant()
    typer.echo(json.dumps(assistant.capabilities(), indent=2))


@app.command()
def run(goal: str, session_id: str | None = None, as_json: bool = False) -> None:
    assistant = AutonomousAssistant()
    report = assistant.run(goal=goal, session_id=session_id)
    if as_json:
        typer.echo(json.dumps(report.model_dump(), indent=2))
        return

    typer.echo(f"Session: {report.session_id}")
    typer.echo(f"Status: {report.status.value}")
    typer.echo("")
    typer.echo(report.final_output)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()

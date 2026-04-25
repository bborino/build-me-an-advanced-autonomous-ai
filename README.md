# Autonomous Assistant

An advanced autonomous AI assistant designed to interpret goals, build plans, route work across tools and models, execute steps, evaluate outcomes, and retry when needed.

## What It Does

- Understands intent and converts vague goals into structured execution plans
- Chooses tools dynamically across web search, HTTP APIs, shell, Python execution, and workspace file operations
- Routes tasks to the best available model profile for planning, execution, evaluation, and synthesis
- Stores session memory and action logs in SQLite so it can reuse relevant context
- Evaluates each task outcome, retries with alternative strategies, and returns a concise final result

## Included Interfaces

- CLI for local runs
- FastAPI server for programmatic use
- SQLite-backed memory store
- Safe-by-default shell and filesystem tools
- Pluggable OpenAI-compatible and Anthropic model providers

## Quick Start

1. Create a virtual environment and install the package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

2. Copy `.env.example` to `.env` and add any API keys you want to use.

3. Run the assistant:

```powershell
autonomous-assistant run "Research the best open-source vector databases and summarize the tradeoffs"
```

4. Or start the API server:

```powershell
autonomous-assistant serve --host 127.0.0.1 --port 8000
```

## Example CLI Commands

```powershell
autonomous-assistant capabilities
autonomous-assistant run "Inspect this project and propose a release plan"
autonomous-assistant run "Search the web for Python 3.14 changes and save a summary in notes/py314.md"
```

## API

### `GET /health`

Returns a simple health payload.

### `GET /capabilities`

Lists registered tools, model profiles, and safety posture.

### `POST /run`

Request body:

```json
{
  "goal": "Find three API providers for company enrichment and compare pricing",
  "session_id": null
}
```

Response: a full run report containing the interpreted intent, task plan, execution steps, final output, and overall status.

## Architecture

The assistant follows this loop:

1. Interpret user goal
2. Build a dependency-aware plan
3. Select the best model and tool for the current task
4. Execute one step
5. Evaluate the result
6. Retry, adapt, or finalize

Core modules:

- `assistant.py`: top-level autonomous execution loop
- `reasoning.py`: goal interpretation, planning, next-action selection, evaluation, synthesis
- `router.py`: multi-model routing and fallback selection
- `memory.py`: SQLite persistence for sessions, notes, and logs
- `tools/`: web, HTTP, filesystem, Python execution, and shell tools
- `providers/`: model connectors

## Safety Model

- Destructive shell commands are blocked
- Sensitive system-changing shell commands require confirmation
- Filesystem writes are restricted to the workspace root
- Filesystem deletion is intentionally not implemented
- The assistant records actions for transparency

## Notes

- The assistant works without API keys using heuristic fallbacks, but model-backed planning and tool selection become much stronger once you configure providers.
- Web search uses Brave Search when `BRAVE_SEARCH_API_KEY` is present, otherwise it falls back to DuckDuckGo HTML parsing and Wikipedia search.


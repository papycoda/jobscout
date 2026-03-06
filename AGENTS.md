# Repository Guidelines

## Project Structure & Module Organization
- `jobscout/`: core domain code. Legacy CLI pipeline lives in `main.py`, `scoring.py`, `filters.py`, and `resume_parser.py`. Agent-based modules live in `agent.py`, `agent_models.py`, `llm_client.py`, `emailer_agent.py`, and `job_sources/fetcher.py`.
- `backend/`: FastAPI service. `backend/main.py` exposes both `/api/search` (agent-backed via `backend/agent_handler.py`) and `/api/search-legacy`.
- `tests/`: primary pytest suite for scoring, job sources, adapters, semantic fallback, and backend behavior.
- Root `test_*.py` files: broader integration/stateless checks (for example `test_stateless_api.py`).
- `scripts/`: developer utilities (`lint.sh`, `preload_semantic_model.py`).
- `docs/plans/`: implementation plans and migration notes (including v2.0 agent rollout).

## Build, Test, and Development Commands
- `pip install -r requirements.txt` and `pip install -r backend/requirements.txt`: install runtime and API dependencies.
- `python jobscout_cli.py --config config.yaml`: run the legacy CLI flow once.
- `python jobscout_cli.py --schedule`: run scheduled legacy execution.
- `python -m uvicorn backend.main:app --reload --port 8000`: run API locally (agent-backed search path).
- `pytest tests/ -v`: run the main test suite.
- `pytest tests/ --cov=jobscout --cov-report=html`: generate coverage output.
- `pytest test_stateless_api.py -v`: run top-level API regression.
- `bash scripts/lint.sh`: lightweight syntax/import checks.

## Coding Style & Naming Conventions
- Use Python defaults: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep docstrings short and behavior-focused.
- Prefer explicit type hints at module/API boundaries.
- Use `logging` for runtime diagnostics; reserve `print` for CLI UX messages.
- When updating agent data contracts, keep `agent_models.py` `to_dict`/`from_dict` behavior consistent.

## Testing Guidelines
- Framework: `pytest`; naming pattern: `test_<feature>.py` and descriptive `test_<behavior>` functions.
- Add coverage for both touched path and fallback path (agent + legacy where relevant).
- Keep LLM-dependent tests skip-safe when provider keys are absent; default test runs should not require live API keys.

## Commit & Pull Request Guidelines
- Follow Conventional Commit prefixes used in history: `feat:`, `fix:`, `refactor:`, `build:`; scoped forms like `feat(job_scoring): ...` are acceptable.
- Keep commits focused and reversible.
- PRs should include purpose, implementation summary, commands run, and sample API request/response for endpoint changes.

## Security & Configuration Tips
- Never commit personal resumes, `config.yaml`, or real secrets.
- Store sensitive values in env vars (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `API_KEY`, `METRICS_API_KEY`, SMTP credentials).
- Keep `CORS_ORIGINS` minimal in deployed environments.

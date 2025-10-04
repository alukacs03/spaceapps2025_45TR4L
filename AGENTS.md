\# Repository Guidelines

## Project Structure & Module Organization
- Root: static frontend (`index.html`, `asteroid.html`).
- `api/`: FastAPI service (`app.py`), `requirements.txt`, local virtual env folder (`venv/`).
- `resources/`: texture and image assets.
- `models/`: 3D assets (e.g., `asteroid.glb`).

## Build, Test, and Development Commands
- Frontend (static):
  - `python3 -m http.server 8000` (from repo root)
  - Open `http://localhost:8000/index.html` (or `asteroid.html`).
- Backend (API):
  - `cd api && python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn app:app --reload --port 8080`
  - Health check: `curl http://localhost:8080/health`
  - Example: `curl 'http://localhost:8080/getPopulation?lat=0&lon=0&radius=50'`
- Env config: create `api/.env` with `GEONAMES_USERNAME=<your_geonames_username>` (already git-ignored).

## Coding Style & Naming Conventions
- Python: 4-space indent, snake_case for functions/variables, PascalCase for classes, type hints for new code.
- Keep functions focused and small; prefer pure helpers where possible.
- FastAPI routes live in `api/app.py`; add new endpoints near related ones or refactor into modules if they grow.
- Use docstrings for public helpers; prefer `logging` for new debug output (existing prints are tolerated).

## Testing Guidelines
- Recommended: `pytest` with FastAPI `TestClient`.
- Place tests under `api/tests/`, named `test_*.py` (e.g., `api/tests/test_population.py`).
- Suggested targets: geometry helpers, error handling, and `/getPopulation` happy-path.
- Run (after adding dev deps): `pytest -q` and consider `pytest-cov` with ~80% target.

## Commit & Pull Request Guidelines
- Current history is informal; adopt Conventional Commits going forward:
  - Examples: `feat(api): add /isOcean endpoint`, `fix(api): handle WorldPop timeout`.
- PRs should include:
  - Purpose and scope, linked issue (if any).
  - Local test steps (commands above) and expected results.
  - UI changes: attach screenshots or short screen capture of the page.
  - Note any config/env variables added or changed.

## Security & Assets
- Never commit secrets; keep them in `api/.env`.
- Large binaries live in `resources/` and `models/`; avoid churn. Consider Git LFS if asset counts grow.

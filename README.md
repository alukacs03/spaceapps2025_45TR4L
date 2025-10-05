# Project Perigee – NEO Impact Simulator

Interactive Near‑Earth Object (NEO) impact exploration tool created for NASA Space Apps Challenge 2025 — Meteor Madness.

- Challenge: https://www.spaceappschallenge.org/2025/challenges/meteor-madness/
- Team: Ákos Lukács, Daniel Brendan Woods, Erik Gasteiger
- Note: Portions of this project were generated with the assistance of AI coding tools.

## Overview

Project Perigee lets you:

- Browse NASA NeoWs close‑approach feeds (today or historical ranges)
- Pick an asteroid or define a custom one (diameter, velocity, composition/density)
- Choose an impact point on a world map
- Launch an interactive 3D impact visualization (globe.gl + three.js)
- Estimate exposed population within ring distances via a lightweight FastAPI backend (WorldPop)

## Repository Structure

- `index.html` – Main UI (NEO feed, custom asteroid, map, launch)
- `asteroid.html` – 3D impact viewer (globe.gl/three.js)
- `api/` – FastAPI backend
  - `app.py` – Endpoints for population queries and impact model summary
  - `requirements.txt` – Python dependencies
  - `.env` – Local env (git‑ignored). Configure `GEONAMES_USERNAME` here
- `resources/` – Texture and image assets (e.g., Earth maps)
- `models/` – 3D assets (e.g., `asteroid.glb`)

## Quick Start

Frontend (static):

1) From repo root, serve the site:

   `python3 -m http.server 8000`

2) Open:

   - `http://localhost:8000/index.html` (main app)
   - `http://localhost:8000/asteroid.html` (3D viewer directly)

Backend (API):

1) Create and activate a venv, then install deps:

   `cd api && python -m venv .venv && source .venv/bin/activate`

   `pip install -r requirements.txt`

2) Configure environment (optional but recommended for `/isOcean`):

   - Create `api/.env` with:

     `GEONAMES_USERNAME=<your_geonames_username>`

3) Run the server:

   `uvicorn app:app --reload --port 8080`

4) Health check:

   `curl http://localhost:8080/health`

Example population query:

`curl 'http://localhost:8080/getPopulation?lat=0&lon=0&radius=50'`

## Using the App

- Enter or keep `DEMO_KEY` for NASA API key (stored locally) and load today’s feed or pick a historical range.
- Select an item from the feed (or use the “Custom asteroid” panel).
- Click on the map to set the impact location. Lat/Lon fields update automatically.
- Click LAUNCH to open the 3D viewer. The scene will pre‑fetch backend data to prepare population summaries.

Notes:

- WebGL: The viewer initializes with a conservative renderer configuration and retry fallback. If initialization fails, ensure WebGL is enabled and try another browser/device.
- Assets: High‑resolution globe textures are in `resources/`. Large binaries are kept in `resources/` and `models/` to minimize churn.

## API

Base URL (dev): `http://localhost:8080`

- `GET /health` – Liveness probe
- `GET /isOcean?lat=<float>&lon=<float>` – Ocean/land heuristic
  - Primary: GeoNames oceanJSON (requires `GEONAMES_USERNAME`)
  - Fallback: OSM Nominatim reverse lookup
- `POST /impact/summary` – Run the impact model and return a summary
  - Body shape:
    - `projectile`: `{ diameter_m, speed_mps, density_kgpm3, angle_deg }`
    - `target`: `{ kind: "crystalline"|"sedimentary"|"water", density_kgpm3?, gravity_mps2 }`
    - `options?`: `{ luminous_efficiency, tsunami_depth_m, tsunami_distance_km, tsunami_slope_ratio }`
- `GET /getPopulation` – Population within a circle of given radius (km)
  - Query: `lat`, `lon`, `radius`, `year` (2000–2020), `dataset` (`wpgppop`|`wpgpas`), optional `api_key`
  - If area exceeds allowance, requests are tiled into pie slices and summed concurrently

## Development

- Coding style
  - Python: 4‑space indent, snake_case, type hints for new code
  - FastAPI routes live in `api/app.py`
- Testing (recommended): `pytest` with FastAPI `TestClient`
  - Place tests under `api/tests/` → `test_*.py`
  - Suggested: geometry helpers, error handling, `/getPopulation` happy path
- Commit messages: Conventional Commits (e.g., `feat(api): add /isOcean endpoint`)

## Troubleshooting

- 3D viewer reports “Error creating WebGL context”
  - Ensure browser supports WebGL and hardware acceleration
  - Try a different browser or disable extensions that block WebGL

- Backend requests to WorldPop time out
  - Try again (service occasionally slow); the code retries and tiles large requests
  - Reduce radius or adjust `max_area_km2_hint`/`allowance_safety` if needed

## Acknowledgements

- NASA Near Earth Object Web Service (NeoWs)
- WorldPop API
- OpenStreetMap Nominatim and GeoNames
- globe.gl and three.js for 3D visualization

## License

Attribution–NonCommercial License (ANCL).

- You may use, copy, modify, and distribute this project for non‑commercial purposes with attribution.
- Commercial use is not permitted without a separate license.

See the LICENSE file for full terms.

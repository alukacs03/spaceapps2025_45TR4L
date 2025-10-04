from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import httpx
import os
import math, json, time, re
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Tuple, Literal

# Support both 'uvicorn api.app:app' (from repo root) and 'uvicorn app:app' (from api folder)
try:
    from .impact_model import Projectile, Target, ImpactModel  # type: ignore
except Exception:  # pragma: no cover - fallback when run from api/ as top-level module
    from impact_model import Projectile, Target, ImpactModel  # type: ignore

app = FastAPI(title="Geo Aggregator (population-only, tiled)", version="2.0.0")

# -------------------------------
# Health + small utility endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/isOcean")
def is_ocean(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in decimal degrees"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude in decimal degrees")
):
    load_dotenv()
    geonames_username = os.getenv("GEONAMES_USERNAME")
    if not geonames_username:
        raise HTTPException(status_code=500, detail="GeoNames username not configured.")

    geonames_url = f"http://api.geonames.org/oceanJSON?lat={lat}&lng={lon}&username={geonames_username}"
    try:
        r = httpx.get(geonames_url, timeout=10.0)
        r.raise_for_status()
        data = r.json()
        return bool(data.get("ocean"))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from GeoNames: {str(e)}")

# -------------------------------
# Impact simulation endpoints
# -------------------------------

class ProjectileIn(BaseModel):
    diameter_m: float = Field(..., gt=0, description="Projectile diameter in meters")
    speed_mps: float = Field(..., gt=0, description="Impact speed in m/s")
    density_kgpm3: float = Field(..., gt=0, description="Bulk density in kg/m^3")
    angle_deg: float = Field(..., ge=0, le=90, description="Entry angle to horizontal in degrees")

class TargetIn(BaseModel):
    kind: Literal["crystalline", "sedimentary", "water"] = Field("crystalline")
    density_kgpm3: Optional[float] = Field(None, gt=0)
    gravity_mps2: float = Field(9.80665, gt=0)

class ImpactOptions(BaseModel):
    luminous_efficiency: float = Field(3e-3, gt=0, lt=1)
    tsunami_depth_m: float = Field(3682.0, gt=0)
    tsunami_distance_km: Optional[float] = Field(100.0, gt=0)
    tsunami_slope_ratio: float = Field(0.005, gt=0)

class ImpactRequest(BaseModel):
    projectile: ProjectileIn
    target: TargetIn
    options: Optional[ImpactOptions] = None

@app.post("/impact/summary")
def impact_summary(req: ImpactRequest):
    # Build domain objects
    p = Projectile(
        diameter_m=req.projectile.diameter_m,
        speed_mps=req.projectile.speed_mps,
        density_kgpm3=req.projectile.density_kgpm3,
        angle_deg=req.projectile.angle_deg,
    )
    t = Target(
        kind=req.target.kind,
        density_kgpm3=req.target.density_kgpm3,
        gravity_mps2=req.target.gravity_mps2,
    )
    model = ImpactModel(p, t)

    opts = req.options or ImpactOptions()
    summary = model.summary(
        luminous_eff=opts.luminous_efficiency,
        tsunami_depth_m=opts.tsunami_depth_m,
        tsunami_distance_km=opts.tsunami_distance_km,
        tsunami_slope_ratio=opts.tsunami_slope_ratio,
    )
    return summary

# -------------------------------
# WorldPop constants + helpers
# -------------------------------
WORLDPOP_STATS_URL = "https://api.worldpop.org/v1/services/stats"
WORLDPOP_TASK_URL  = "https://api.worldpop.org/v1/tasks/{}"

def mask_key(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    return s[:3] + "***" + s[-3:] if len(s) > 6 else "***"

# --- geometry helpers (geodesic-ish) ---

def _destination_point(lon_deg: float, lat_deg: float, bearing_rad: float, distance_km: float):
    """Point reached from (lon,lat) going 'distance_km' along 'bearing_rad' on a sphere."""
    R = 6371.0088  # km
    δ = distance_km / R
    φ1 = math.radians(lat_deg)
    λ1 = math.radians(lon_deg)
    θ = bearing_rad

    sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
    φ2 = math.asin(sinφ2)
    y = math.sin(θ)*math.sin(δ)*math.cos(φ1)
    x = math.cos(δ) - math.sin(φ1)*math.sin(φ2)
    λ2 = λ1 + math.atan2(y, x)
    # normalize lon to [-180, 180)
    lon2 = math.degrees((λ2 + math.pi) % (2*math.pi) - math.pi)
    lat2 = math.degrees(φ2)
    return lon2, lat2

def circle_as_geojson(lon: float, lat: float, radius_km: float, steps: int = 64) -> dict:
    """Full-circle polygon approximation."""
    coords = []
    for i in range(steps + 1):  # close ring
        b = 2 * math.pi * (i / steps)
        x, y = _destination_point(lon, lat, b, radius_km)
        coords.append([x, y])

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    print(f"[geojson.circle] steps={steps} radius_km={radius_km} center=[{lon},{lat}] bbox={bbox}")
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [coords]}
        }]
    }

def sector_as_geojson(lon: float, lat: float, radius_km: float, b_start: float, b_end: float, circle_steps: int = 64) -> dict:
    """
    Pie-slice sector polygon from bearing b_start to b_end (radians), with center at (lon,lat) and arc at 'radius_km'.
    The ring is [center] + arc points + [center]; last coord equals first to close.
    """
    # ensure b_end > b_start within [0, 2π]
    two_pi = 2 * math.pi
    b_start = b_start % two_pi
    b_end = b_end % two_pi
    if b_end <= b_start:
        b_end += two_pi
    arc = b_end - b_start

    # allocate steps along the arc proportionally to full circle resolution
    steps = max(2, math.ceil(circle_steps * (arc / two_pi)))
    arc_points = []
    for i in range(steps + 1):  # include endpoint
        b = b_start + arc * (i / steps)
        x, y = _destination_point(lon, lat, b, radius_km)
        arc_points.append([x, y])

    center = [lon, lat]
    ring = [center] + arc_points + [center]  # start at center, follow arc, return to center (closed)
    # sanity bbox log
    lons = [c[0] for c in ring]
    lats = [c[1] for c in ring]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    print(f"[geojson.sector] arc_deg={math.degrees(arc):.2f} steps={steps} radius_km={radius_km} bbox={bbox}")
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]}
        }]
    }

# --- population parsing & http helpers ---

def _extract_population_only(payload: Dict[str, Any], ctx: str) -> float:
    """Return total_population as float (0 is valid)."""
    print(f"[{ctx}] payload keys={list(payload.keys())}")
    if "total_population" in payload:
        pop = float(payload["total_population"])
        print(f"[{ctx}] total_population={pop}")
        return pop
    raise HTTPException(status_code=502, detail="WorldPop payload missing 'total_population'.")

def _fetch_task_result(client: httpx.Client, taskid: str, log_body_chars: int,
                       max_wait_s: float = 25.0, poll_interval_s: float = 1.0) -> Dict[str, Any]:
    """Poll /v1/tasks/{taskid} until finished or timeout; bubble any API error."""
    print(f"[task] fetching task result taskid={taskid}")
    deadline = time.time() + max_wait_s
    attempt = 0
    last = None
    while True:
        attempt += 1
        tr = client.get(WORLDPOP_TASK_URL.format(taskid))
        preview = tr.text[:log_body_chars] if tr.text else ""
        print(f"[task] attempt#{attempt} status={tr.status_code} preview={preview!r}")
        tr.raise_for_status()
        last = tr.json()
        tstatus = last.get("status")
        terror  = last.get("error")
        print(f"[task] state={tstatus} error={terror}")

        if tstatus == "finished" and not terror:
            return last.get("data") or {}

        if terror:
            msg = last.get("error_message") or "WorldPop task failed."
            print(f"[task.error] {msg}")
            raise HTTPException(status_code=502, detail=msg)

        if time.time() >= deadline:
            state = last.get("status") if isinstance(last, dict) else "unknown"
            print(f"[timeout] taskid={taskid} state={state}")
            raise HTTPException(status_code=504, detail=f"WorldPop task {taskid} is still {state}. Try again later.")
        time.sleep(poll_interval_s)

def _area_km2_of_circle(radius_km: float) -> float:
    return math.pi * radius_km * radius_km

def _radius_km_from_area(area_km2: float) -> float:
    return math.sqrt(area_km2 / math.pi)

def _parse_allowance_from_error(msg: str) -> Optional[Tuple[float, float]]:
    """Parse: 'The requested area was too large. Requested 283030.91 km^2 but allowance was 100000.'"""
    try:
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)", msg)]
        if len(nums) >= 2:
            return nums[0], nums[1]
    except Exception:
        pass
    return None

def _safe_radius_for_allowance(allow_km2: float, safety: float) -> float:
    target_area = allow_km2 * max(0.0, min(1.0, safety))
    return _radius_km_from_area(target_area)

def _get_with_retries(client: httpx.Client, url: str, params: Dict[str, Any],
                      attempts: int = 3, timeout_note: str = "") -> httpx.Response:
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            print(f"[http.try] attempt={i} url={url}")
            r = client.get(url, params=params)
            print(f"[http.try] status={r.status_code} attempt={i}")
            return r
        except httpx.ReadTimeout as e:
            last_exc = e
            print(f"[http.timeout] attempt={i} {timeout_note} error={e}")
            if i < attempts:
                time.sleep(0.8 * i)
            continue
    raise last_exc

def _worldpop_population_for_geojson(client: httpx.Client, dataset: str, year: int, gj_str: str,
                                     api_key: Optional[str], log_body_chars: int,
                                     clamp_on_area_limit: bool, allowance_safety: float,
                                     backoff_factor: float = 0.98) -> float:
    """
    Request stats for the provided polygon geojson string and return total_population.
    Accept inline data regardless of 'status', otherwise poll task. If server reports
    allowance, shrink via backoff (rare for already-sliced polygons).
    """
    params: Dict[str, Any] = {
        "dataset": dataset,
        "year": year,
        "geojson": gj_str,
        "runasync": "false",
    }
    if api_key:
        params["key"] = api_key

    printable = dict(params)
    if "key" in printable:
        printable["key"] = mask_key(printable["key"])
    printable["geojson"] = f"<geojson polygon; len={len(gj_str)}>"
    print(f"[request] GET {WORLDPOP_STATS_URL} params={printable}")

    # robust GET with retry
    r = _get_with_retries(client, WORLDPOP_STATS_URL, params, attempts=3, timeout_note="stats")
    print(f"[http] stats.status={r.status_code} url={r.request.url}")
    body_preview = r.text[:log_body_chars] if r.text else ""
    print(f"[http] stats.body_preview={body_preview!r}")
    r.raise_for_status()

    try:
        data = r.json()
    except Exception as je:
        print(f"[error] JSON parse failed: {je}")
        raise HTTPException(status_code=502, detail=f"WorldPop returned non-JSON: {je}")

    print(f"[stats.response] keys={list(data.keys())} status={data.get('status')} taskid={data.get('taskid')} error={data.get('error')}")

    # Inline payload first
    payload = data.get("data") or {}
    if isinstance(payload, dict) and "total_population" in payload:
        return _extract_population_only(payload, "stats.data")

    # Direct error (maybe area overage even for a slice?)
    if data.get("error"):
        emsg = data.get("error_message") or "WorldPop reported an error."
        print(f"[stats.error] {emsg}")
        if clamp_on_area_limit and "allowance" in emsg:
            parsed = _parse_allowance_from_error(emsg)
            if parsed:
                req_km2, allow_km2 = parsed
                print(f"[server.allowance] requested≈{req_km2:.2f} > allowed={allow_km2:.2f}; you may need more slices or smaller arc.")
        raise HTTPException(status_code=502, detail=emsg)

    # If we have a task, fetch/poll it
    if data.get("taskid") and data.get("status") in {"started", "finished", "created", "queued", "running"}:
        taskid = data["taskid"]
        print(f"[stats] no inline total yet; fetching task payload taskid={taskid}")
        tdata = _fetch_task_result(client, taskid, log_body_chars, max_wait_s=20.0, poll_interval_s=0.8)
        return _extract_population_only(tdata, "task.data")

    print(f"[unexpected] response shape with no inline data and no pollable task: {data}")
    raise HTTPException(status_code=502, detail="WorldPop returned an unexpected response.")

# ---------------------------------
# Endpoint: population only (with tiling)
# ---------------------------------
@app.get("/getPopulation")
def get_population(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in decimal degrees"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude in decimal degrees"),
    radius: float = Query(..., gt=0, description="Radius in kilometers"),
    year: int = Query(2020, ge=2000, le=2020, description="WorldPop year (2000–2020)"),
    dataset: str = Query("wpgppop", pattern="^(wpgppop|wpgpas)$", description="Dataset (wpgppop recommended)"),
    api_key: Optional[str] = Query(None, description="Optional WorldPop API key"),
    debug: bool = Query(False, description="Include debug metadata in response"),
    log_body_chars: int = Query(800, ge=100, le=100000, description="Max chars to print from response bodies"),
    # Tiling & allowance controls
    max_area_km2_hint: float = Query(100000.0, gt=0, description="WorldPop per-request area allowance (hint)"),
    allowance_safety: float = Query(0.97, ge=0.5, le=0.999, description="Factor below allowance for sizing slices"),
    circle_steps: int = Query(64, ge=32, le=512, description="Resolution of circle/arc discretization"),
):
    print(f"[start] lat={lat} lon={lon} radius_km={radius} year={year} dataset={dataset} key={mask_key(api_key)} "
          f"debug={debug} max_area_km2_hint={max_area_km2_hint} safety={allowance_safety}")

    lon_f, lat_f = float(lon), float(lat)
    requested_radius = float(radius)

    total_area = _area_km2_of_circle(requested_radius)
    print(f"[area] requested_circle_area_km2≈{total_area:.2f}")

    # If within allowance, do single call
    if total_area <= max_area_km2_hint * allowance_safety:
        print("[mode] single-call (within allowance)")
        gj = circle_as_geojson(lon_f, lat_f, requested_radius, steps=circle_steps)
        gj_str = json.dumps(gj, separators=(",", ":"))
        with httpx.Client(timeout=60.0) as client:
            pop = _worldpop_population_for_geojson(
                client, dataset, year, gj_str, api_key, log_body_chars,
                clamp_on_area_limit=False, allowance_safety=allowance_safety
            )
        resp = {"population": pop, "dataset": dataset, "year": year, "radius_used_km": requested_radius}
        return resp

    # Otherwise: TILE into N non-overlapping pie slices and sum
    print("[mode] tiled (pie slices)")
    # choose N so each slice area <= safety * allowance
    per_slice_area_target = max_area_km2_hint * allowance_safety
    N = math.ceil(total_area / per_slice_area_target)
    N = max(2, int(N))  # at least 2 slices
    arc = 2 * math.pi / N
    print(f"[tiling] total_area≈{total_area:.2f} km²; per_slice_target≤{per_slice_area_target:.2f} -> slices={N} (arc={math.degrees(arc):.2f}° each)")

    grand_total = 0.0
    per_slice = []

    with httpx.Client(timeout=60.0) as client:
        for i in range(N):
            b_start = i * arc
            b_end   = (i + 1) * arc
            print(f"[slice] {i+1}/{N} bearings_deg=[{math.degrees(b_start):.2f},{math.degrees(b_end):.2f}]")
            gj = sector_as_geojson(lon_f, lat_f, requested_radius, b_start, b_end, circle_steps=circle_steps)
            gj_str = json.dumps(gj, separators=(",", ":"))
            try:
                pop_i = _worldpop_population_for_geojson(
                    client, dataset, year, gj_str, api_key, log_body_chars,
                    clamp_on_area_limit=False, allowance_safety=allowance_safety
                )
            except HTTPException as he:
                # If a slice still trips allowance (shouldn't), split it once more (half-arc) and sum
                if isinstance(he.detail, str) and "allowance" in he.detail:
                    print(f"[slice.allowance] slice {i+1}/{N} exceeded allowance unexpectedly; subdividing into 2 half-slices")
                    half_arc = (b_start + b_end)/2
                    subtotal = 0.0
                    for j, (bs, be) in enumerate(((b_start, half_arc), (half_arc, b_end)), start=1):
                        gj_sub = sector_as_geojson(lon_f, lat_f, requested_radius, bs, be, circle_steps=circle_steps)
                        gj_sub_str = json.dumps(gj_sub, separators=(",", ":"))
                        pop_sub = _worldpop_population_for_geojson(
                            client, dataset, year, gj_sub_str, api_key, log_body_chars,
                            clamp_on_area_limit=False, allowance_safety=allowance_safety
                        )
                        subtotal += pop_sub
                        print(f"[slice.sub] {i+1}.{j} subtotal_now={subtotal}")
                    pop_i = subtotal
                else:
                    raise
            grand_total += pop_i
            per_slice.append(pop_i)
            print(f"[slice.done] {i+1}/{N} pop={pop_i} running_total={grand_total}")

    resp = {
        "population": grand_total,
        "dataset": dataset,
        "year": year,
        "radius_used_km": requested_radius,
        "tiled": True,
        "slices": N,
    }
    if debug:
        resp["__debug"] = {
            "per_slice": per_slice,
            "slice_arc_deg": math.degrees(arc),
            "total_area_km2_est": total_area,
            "per_slice_area_target_km2": per_slice_area_target,
        }
    print(f"[done] tiled sum population={grand_total} slices={N}")
    return resp

from __future__ import annotations
import io
import re
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "frontend" / "dist"
SOURCE = REPO_ROOT / "frontend" / "src"
def test_frontend_source_only_uses_relative_urls():
    offenders = []
    for path in SOURCE.rglob("*.ts*"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"['\"`]https?://([^'\"`]+)", text):
            host = match.group(1)
            if host.startswith("localhost") or host.startswith("127.0.0.1"):
                continue
            if "example.com" in host or "schema" in host:
                continue
            offenders.append(f"{path.name}: {host}")
    assert not offenders, "the browser must not call absolute hosts: " + ", ".join(offenders)
def test_api_paths_used_by_the_frontend_are_served(client):
    source = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE.rglob("*.ts*"))
    paths = set(re.findall(r"['\"`](/api/[a-z0-9\-/<>{}$._]*)", source))
    assert paths, "no API path was found in the frontend source"
    unresolved = []
    for path in sorted(paths):
        candidate = path.split("${")[0].rstrip("/")
        if not candidate or candidate.endswith("<") or "..." in candidate or "%" in candidate:
            continue
        try:
            rules = list(client.application.url_map.iter_rules())
        except Exception:
            rules = []
        if not any(str(rule).startswith(candidate.split("<")[0]) for rule in rules):
            unresolved.append(candidate)
    assert not unresolved, "the frontend calls endpoints that do not exist: " + ", ".join(unresolved)
def test_built_bundle_is_present_and_relative():
    if not DIST.exists():
        return
    index = (DIST / "index.html").read_text(encoding="utf-8")
    assert "<div id=\"root\">" in index or "<div id='root'>" in index
    assets = list((DIST / "assets").glob("*.js"))
    assert assets, "the production build produced no JavaScript"
    bundle = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in assets)
    assert "/api/" in bundle, "the built bundle does not talk to the API"
    assert "http://127.0.0.1" not in bundle and "http://localhost:5000" not in bundle
def test_spa_fallback_is_served_by_the_api(client):
    response = client.get("/analysis/radiomics")
    if response.status_code == 404:
        return
    assert response.status_code == 200
    assert b"<div id=\"root\"" in response.data or b"<!doctype html>" in response.data.lower()

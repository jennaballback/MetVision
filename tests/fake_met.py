"""A local stand-in for the Met Collection API and image host.

Serves Met-shaped JSON for a few hundred synthetic objects plus generated JPEGs,
so the whole pipeline can run offline (tests, CI, sandboxes without network).
Object images are coloured by classification with noise, so a visual embedder
finds real structure and the eval has something to measure.

    python -m tests.fake_met --port 8765      # then MET_API_BASE=http://localhost:8765/v1
"""
from __future__ import annotations

import argparse
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw

DEPARTMENTS = [
    {"departmentId": 6, "displayName": "Asian Art"},
    {"departmentId": 9, "displayName": "Drawings and Prints"},
    {"departmentId": 10, "displayName": "Egyptian Art"},
    {"departmentId": 11, "displayName": "European Paintings"},
]
DEPT_PROFILES = {
    6: [("Japan", "Prints", "Edo period (1615–1868)", (1700, 1868), (200, 60, 50)),
        ("China", "Paintings", "Qing dynasty (1644–1911)", (1650, 1900), (90, 140, 90)),
        ("China", "Ceramics", "Ming dynasty (1368–1644)", (1368, 1644), (60, 90, 200))],
    9: [("French", "Prints", "", (1800, 1900), (40, 40, 40)),
        ("Dutch", "Drawings", "", (1600, 1700), (210, 190, 150))],
    10: [("Egyptian", "Sculpture", "New Kingdom", (-1550, -1070), (190, 160, 90))],
    11: [("French", "Paintings", "", (1850, 1900), (120, 160, 210)),
         ("Italian", "Paintings", "", (1400, 1550), (150, 100, 40))],
}
SUBJECTS = ["Bird", "Landscape", "Portrait", "Flowers", "River", "Mountain", "Woman Reading", "Horse"]
TAGS = ["Birds", "Landscapes", "Portraits", "Flowers", "Rivers", "Mountains", "Women", "Horses"]


def build_catalog(per_department: int = 120, seed: int = 1) -> dict[int, dict]:
    rng = random.Random(seed)
    catalog = {}
    oid = 1000
    for dept in DEPARTMENTS:
        did = dept["departmentId"]
        for _ in range(per_department):
            oid += rng.randint(1, 3)
            culture, classification, period, (y0, y1), _ = rng.choice(DEPT_PROFILES[did])
            s = rng.randrange(len(SUBJECTS))
            begin = rng.randint(y0, y1 - 10)
            has_image = rng.random() > 0.15
            public = rng.random() > 0.1
            catalog[oid] = {
                "objectID": oid, "isHighlight": False, "accessionNumber": f"F.{oid}",
                "isPublicDomain": public,
                "primaryImage": f"/img/{oid}.jpg" if has_image and public else "",
                "primaryImageSmall": f"/img/{oid}.jpg" if has_image and public else "",
                "additionalImages": [], "constituents": None,
                "department": dept["displayName"], "objectName": classification.rstrip("s"),
                "title": f"{SUBJECTS[s]} No. {oid}", "culture": culture, "period": period,
                "dynasty": "", "reign": "", "portfolio": "", "artistRole": "",
                "artistDisplayName": rng.choice(["", "Unknown Artist", f"Artist {oid % 17}"]),
                "objectDate": rng.choice([f"ca. {begin}", f"{begin}–{begin + 10}", "late period"]),
                "objectBeginDate": begin, "objectEndDate": begin + rng.randint(0, 10),
                "medium": rng.choice(["Ink and color on paper", "Oil on canvas", "Limestone", "Porcelain"]),
                "classification": classification,
                "objectURL": f"https://www.metmuseum.org/art/collection/search/{oid}",
                "tags": ([{"term": TAGS[s], "AAT_URL": "", "Wikidata_URL": ""}] if rng.random() > 0.4 else None),
                "metadataDate": "2026-01-15T04:00:00Z", "GalleryNumber": "",
                "_subject": s,
            }
    return catalog


def render_image(obj: dict) -> bytes:
    """Base colour from (culture, classification) plus a subject-specific shape."""
    did = next(d["departmentId"] for d in DEPARTMENTS if d["displayName"] == obj["department"])
    color = next(p[4] for p in DEPT_PROFILES[did]
                 if p[0] == obj["culture"] and p[1] == obj["classification"])
    rng = random.Random(obj["objectID"])
    jitter = tuple(max(0, min(255, c + rng.randint(-35, 35))) for c in color)
    im = Image.new("RGB", (160, 200), jitter)
    d = ImageDraw.Draw(im)
    s = obj["_subject"]
    x, y = 30 + s * 10, 40 + rng.randint(0, 40)
    d.ellipse([x, y, x + 60, y + 60], fill=((s * 31) % 255, (s * 67) % 255, (s * 97) % 255))
    buf = BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


class FakeMet:
    def __init__(self, per_department: int = 120, port: int = 0):
        self.catalog = build_catalog(per_department)
        self.withdrawn: set[int] = set()       # ids that now 404
        self.updated_since: dict[int, str] = {}  # id -> date it was "updated"
        self.requests = 0
        self.fail_next = 0                      # inject transient errors...
        self.fail_status = 503                  # ...with this status
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()

    def public(self, obj: dict) -> dict:
        out = {k: v for k, v in obj.items() if not k.startswith("_")}
        for key in ("primaryImage", "primaryImageSmall"):
            if out[key]:
                out[key] = self.base + out[key]
        return out

    def add_object(self, department_id: int, **overrides) -> int:
        oid = max(self.catalog) + 5
        template = next(o for o in self.catalog.values()
                        if o["department"] == next(d["displayName"] for d in DEPARTMENTS
                                                   if d["departmentId"] == department_id) and o["primaryImageSmall"])
        self.catalog[oid] = {**template, "objectID": oid, "title": f"New accession {oid}",
                             "primaryImage": f"/img/{oid}.jpg", "primaryImageSmall": f"/img/{oid}.jpg",
                             "metadataDate": "2026-10-01T04:00:00Z", **overrides}
        self.updated_since[oid] = "2026-10-01"
        return oid

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                fake.requests += 1
                if fake.fail_next > 0:
                    fake.fail_next -= 1
                    return self._json(fake.fail_status, {"message": "try later"})
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                if u.path == "/v1/departments":
                    return self._json(200, {"departments": DEPARTMENTS})
                if u.path == "/v1/search":  # retired by the Met on 2026-10-01
                    return self._json(410, {"message": "/public/collection/v1/search was retired on 2026-10-01."})
                if u.path == "/v1/objects":
                    did = int(q.get("departmentIds", 0))
                    name = next((d["displayName"] for d in DEPARTMENTS if d["departmentId"] == did), None)
                    ids = [o["objectID"] for o in fake.catalog.values()
                           if (name is None or o["department"] == name) and o["objectID"] not in fake.withdrawn]
                    if "metadataDate" in q:
                        ids = [i for i in ids if fake.updated_since.get(i, "0000") >= q["metadataDate"]]
                    return self._json(200, {"total": len(ids), "objectIDs": sorted(ids) or None})
                if u.path.startswith("/v1/objects/"):
                    oid = int(u.path.rsplit("/", 1)[1])
                    if oid not in fake.catalog or oid in fake.withdrawn:
                        return self._json(404, {"message": "ObjectID not found"})
                    return self._json(200, fake.public(fake.catalog[oid]))
                if u.path.startswith("/img/"):
                    oid = int(u.path.rsplit("/", 1)[1].split(".")[0])
                    data = render_image(fake.catalog[oid])
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self._json(404, {"message": "not found"})

        return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--per-department", type=int, default=300)
    a = ap.parse_args()
    f = FakeMet(a.per_department, a.port)
    print(f"fake Met API on {f.base}/v1 ({len(f.catalog)} objects)")
    f.server.serve_forever()

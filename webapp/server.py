"""HTTP layer for the course audit.

Endpoints (all JSON responses use the envelope ``{success, data, error}``):

    GET  /                        the single-page app
    GET  /api/health              liveness and library version
    GET  /api/scenarios           the built-in synthetic scenarios
    GET  /api/scenarios/{id}      one scenario's course and track
    POST /api/parse               upload a GPX / GeoJSON / CSV file -> points
    POST /api/audit               audit a track against a course
    POST /api/gpx                 turn a list of points into a GPX download

Hardening, proportionate to a tool meant to run on localhost: request bodies
are capped by counting the bytes actually received (a chunked body cannot slip
past a missing Content-Length); every input is validated before any
computation; each audit's work is bounded in size and in duration (inputs are
simplified or decimated under explicit work budgets) and at most two run at
once, so a flood gets HTTP 503 rather than an unresponsive machine; browsers
may not POST here from another site; responses carry a strict
Content-Security-Policy; internal errors are logged but never echoed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import frechet_edit

from . import scenarios
from .audit import MAX_DELTA_M, MIN_DELTA_M, AuditInputError, run_audit
from .geo import MAX_POINTS_PER_FILE, Fix, GeoInputError, parse_track_file, to_gpx

log = logging.getLogger("webapp")

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
#: Any request body, counted as it arrives: a 5 MB file plus multipart framing,
#: or an audit of two maximal curves as JSON.
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_CONCURRENT_AUDITS = 2
#: `Sec-Fetch-Site` values a browser sends for requests this page itself made.
#: Non-browser clients send no such header and are unaffected.
ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})

CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https://tile.openstreetmap.org; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'; "
    "object-src 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

_audit_slots = threading.BoundedSemaphore(MAX_CONCURRENT_AUDITS)


class BodySizeLimit:
    """Refuse any request body larger than ``max_bytes``, however it is framed.

    A declared ``Content-Length`` over the limit is refused before a byte is
    read. Otherwise the bytes are counted as the application receives them, so
    a chunked body with no ``Content-Length`` is cut off at the limit instead
    of being buffered whole. (Security review: a 150 MB chunked body once got
    through a Content-Length-only check.)
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _send_json(send, 413, "request body too large")
            return
        received = 0
        exceeded = False
        answered = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            # Once the limit is hit, whatever the inner app makes of the
            # aborted read (FastAPI reports it as a 400) is replaced by a 413.
            nonlocal answered
            if exceeded:
                if message["type"] == "http.response.start" and not answered:
                    answered = True
                    await _send_json(send, 413, "request body too large")
                return
            if message["type"] == "http.response.start":
                answered = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except _BodyTooLarge:
            if not answered:
                await _send_json(send, 413, "request body too large")


class _BodyTooLarge(Exception):
    """Raised inside ``receive`` to stop reading an oversized body."""


async def _send_json(send: Send, status: int, message: str) -> None:
    response = fail(status, message)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response.body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": response.body})


def ok(data: Any) -> dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def fail(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"success": False, "data": None, "error": message}
    )


class FixIn(BaseModel):
    lat: float
    lon: float
    time: str | None = Field(default=None, max_length=40)

    def to_fix(self) -> Fix:
        return Fix(self.lat, self.lon, self.time)


class AuditRequest(BaseModel):
    course: list[FixIn] = Field(min_length=2, max_length=MAX_POINTS_PER_FILE)
    track: list[FixIn] = Field(min_length=2, max_length=MAX_POINTS_PER_FILE)
    delta_m: float = Field(ge=MIN_DELTA_M, le=MAX_DELTA_M)


class GpxRequest(BaseModel):
    name: str = Field(default="track", max_length=100)
    points: list[FixIn] = Field(min_length=1, max_length=MAX_POINTS_PER_FILE)


def _fix_dict(fix: Fix) -> dict[str, Any]:
    return {"lat": fix.lat, "lon": fix.lon, "time": fix.time}


def create_app() -> FastAPI:
    app = FastAPI(
        title="Course audit",
        version=frechet_edit.__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next: Any) -> Response:
        # Multipart uploads are CORS "simple" requests, so without this another
        # site could make a visitor's browser POST files here unseen. Browsers
        # say where a request came from; clients that send nothing (curl,
        # scripts) are not browsers and cannot be driven by a web page.
        site = request.headers.get("sec-fetch-site")
        if request.method == "POST" and site is not None and site not in ALLOWED_FETCH_SITES:
            response: Response = fail(403, "cross-site requests are not accepted")
        else:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return fail(422, f"invalid request: {where or 'body'}: {first.get('msg', 'invalid')}")

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return fail(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return fail(500, "internal error; see the server log")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return ok({"status": "ok", "frechet_edit": frechet_edit.__version__})

    @app.get("/api/scenarios")
    def list_scenarios() -> dict[str, Any]:
        return ok(scenarios.catalogue())

    @app.get("/api/scenarios/{scenario_id}")
    def get_scenario(scenario_id: str) -> Any:
        try:
            sc = scenarios.get(scenario_id)
        except KeyError:
            return fail(404, f"no scenario called {scenario_id!r}")
        return ok(
            {
                "id": sc.id,
                "title": sc.title,
                "blurb": sc.blurb,
                "expectation": sc.expectation,
                "delta_m": sc.delta_m,
                "course": [_fix_dict(f) for f in sc.course],
                "track": [_fix_dict(f) for f in sc.track],
                "injected": list(sc.injected),
            }
        )

    @app.post("/api/parse")
    async def parse(file: Annotated[UploadFile, File()]) -> Any:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            return fail(413, f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        try:
            fixes = parse_track_file(data, file.filename or "upload")
        except GeoInputError as exc:
            return fail(400, str(exc))
        if not fixes:
            return fail(400, "the file contains no points")
        return ok({"name": file.filename, "points": [_fix_dict(f) for f in fixes]})

    @app.post("/api/audit")
    def audit(req: AuditRequest) -> Any:
        if not _audit_slots.acquire(blocking=False):
            return fail(503, "the server is busy with other audits; try again shortly")
        try:
            result = run_audit(
                [f.to_fix() for f in req.course], [f.to_fix() for f in req.track], req.delta_m
            )
        except (GeoInputError, AuditInputError) as exc:
            return fail(400, str(exc))
        finally:
            _audit_slots.release()
        return ok(result)

    @app.post("/api/gpx")
    def gpx(req: GpxRequest) -> Response:
        body = to_gpx([p.to_fix() for p in req.points], req.name)
        safe = "".join(ch for ch in req.name if ch.isalnum() or ch in "-_") or "track"
        return Response(
            content=body,
            media_type="application/gpx+xml",
            headers={"Content-Disposition": f'attachment; filename="{safe}.gpx"'},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)
    return app


app = create_app()

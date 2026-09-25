"""Browser end-to-end tests: a real server, a real browser, the real CSP.

Opt-in, because they need Playwright and a Chromium-family browser:

    pytest -m e2e webapp/tests/test_e2e.py

The browser channel defaults to Microsoft Edge (present on Windows); set
``E2E_BROWSER_CHANNEL=chrome``, or ``chromium`` after ``playwright install
chromium``. If Playwright or the browser is missing these tests are SKIPPED
with the reason, never passed.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

playwright_api = pytest.importorskip("playwright.sync_api")
uvicorn = pytest.importorskip("uvicorn")

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
CHANNEL = os.environ.get("E2E_BROWSER_CHANNEL", "msedge")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def base_url():
    from webapp.server import create_app

    port = _free_port()
    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started:
        if time.time() > deadline:
            pytest.fail("test server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            launched = (
                p.chromium.launch(headless=True)
                if CHANNEL == "chromium"
                else p.chromium.launch(channel=CHANNEL, headless=True)
            )
        except Exception as exc:  # the browser is simply not installed here
            pytest.skip(f"cannot launch browser channel {CHANNEL!r}: {exc}")
        yield launched
        launched.close()


@pytest.fixture
def page(browser, base_url):
    context = browser.new_context(viewport={"width": 1440, "height": 950}, accept_downloads=True)
    page = context.new_page()
    problems: list[str] = []
    page.on("pageerror", lambda exc: problems.append(f"page error: {exc}"))

    def on_console(msg):
        # Network failures are judged by `on_response` below, which knows the
        # URL; the browser's generic "Failed to load resource" line does not.
        if msg.type == "error" and not msg.text.startswith("Failed to load resource"):
            problems.append(f"console {msg.type}: {msg.text}")

    def on_response(response):
        # API errors are deliberate (validation); a failing page asset is a bug.
        # Map tiles need the internet and are not the app's to serve.
        url = response.url
        if response.status >= 400 and "/api/" not in url and "openstreetmap" not in url:
            problems.append(f"asset {url} answered {response.status}")

    page.on("console", on_console)
    page.on("response", on_response)
    page.goto(base_url)
    page.wait_for_selector(".scenario")
    yield page
    context.close()
    assert not problems, problems


def _run_scenario(page, scenario_id: str) -> None:
    page.click(f'.scenario[data-id="{scenario_id}"]')
    page.wait_for_selector("#run:not([disabled])")
    page.click("#run")
    page.wait_for_selector(".stamp", timeout=60_000)


def test_urban_canyon_finds_the_six_spikes(page):
    _run_scenario(page, "urban-canyon")
    assert "On course" in page.inner_text(".stamp")
    assert "6 gps glitches removed" in page.inner_text(".stamp").lower()
    assert page.locator(".ledger tbody tr").count() == 6
    assert page.locator("#trace .trace-glitch").count() == 6
    assert "Independently verified" in page.inner_text(".provenance")


def test_one_lap_of_two_is_caught_although_naive_checks_pass(page):
    _run_scenario(page, "one-lap-of-two")
    assert "Off course" in page.inner_text(".stamp")
    rows = page.locator(".compare tbody tr")
    assert "pass" in rows.nth(0).inner_text() and "pass" in rows.nth(1).inner_text()
    assert "fail" in rows.nth(2).inner_text()


def test_uploading_files_runs_the_same_audit(page):
    page.set_input_files("#course-file", str(SAMPLES / "marina-bay-course.gpx"))
    page.wait_for_selector("#course-drop.is-loaded")
    page.set_input_files("#track-file", str(SAMPLES / "urban-canyon-track.csv"))
    page.wait_for_selector("#track-drop.is-loaded")
    assert "points" in page.inner_text("#track-status")
    page.click("#run")
    page.wait_for_selector(".stamp", timeout=60_000)
    assert page.locator(".ledger tbody tr").count() == 6


def test_a_bad_upload_is_reported_in_place(page, tmp_path):
    bad = tmp_path / "broken.gpx"
    bad.write_text("<gpx><trkpt lat='1'", encoding="utf-8")
    page.set_input_files("#course-file", str(bad))
    page.wait_for_selector("#course-drop.is-error")
    assert "well-formed" in page.inner_text("#course-status")
    assert page.locator("#run").is_disabled()


def test_the_trace_is_keyboard_navigable(page):
    _run_scenario(page, "urban-canyon")
    page.focus("#trace")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    assert page.inner_text("#trace-readout").startswith("fix #1")


def test_cleaned_track_downloads_as_gpx(page):
    _run_scenario(page, "urban-canyon")
    with page.expect_download() as info:
        page.click("text=Cleaned track · GPX")
    content = Path(info.value.path()).read_text(encoding="utf-8")
    assert content.count("<trkpt") == 439 - 6


def test_the_tolerance_slider_updates_the_readout(page):
    page.click('.scenario[data-id="clean"]')
    # Loading a scenario applies its recommended tolerance; wait for that first.
    page.wait_for_selector("#run:not([disabled])")
    page.fill("#delta", "40")
    # The readout separates number and unit with a no-break space; split()
    # treats that as whitespace.
    assert " ".join(page.inner_text("#delta-out").split()) == "±40 m"


def test_phone_width_has_no_horizontal_scroll(browser, base_url):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto(base_url)
    page.wait_for_selector(".scenario")
    _run_scenario(page, "shortcut")
    scroll_width, inner_width = page.evaluate(
        "() => [document.documentElement.scrollWidth, window.innerWidth]"
    )
    context.close()
    assert scroll_width <= inner_width

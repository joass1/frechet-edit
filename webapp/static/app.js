// Course Check front end. One module, no framework, no build step.
// Every piece of user-supplied text (file names, timestamps) is inserted with
// textContent via `h`, never as HTML.

const $ = (selector, root = document) => root.querySelector(selector);
const SVG_NS = "http://www.w3.org/2000/svg";

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function s(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Leaflet renders string tooltips as HTML; handing it a text-only element
// means no tooltip can ever interpret markup, whatever ends up in it.
const tip = (text) => h("span", {}, text);

const state = {
  scenarioId: null,
  course: null,
  track: null,
  courseLabel: null,
  trackLabel: null,
  delta: 25,
  result: null,
  busy: false,
};

// ------------------------------------------------------------------ API

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error("cannot reach the server - is it still running?");
  }
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`the server answered ${response.status} without a readable body`);
  }
  if (!body.success) throw new Error(body.error || `request failed (${response.status})`);
  return body.data;
}

const postJson = (path, payload) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

function setStatus(message, isError = false) {
  const el = $("#status");
  el.textContent = message;
  el.classList.toggle("is-error", isError);
}

// ------------------------------------------------------------------ map

const mapState = { map: null, base: null, overlay: null, corridor: null, hover: null, glitchMarkers: new Map() };

function initMap() {
  if (!window.L) {
    $("#map").append(h("p", { class: "fine" }, "The map library did not load; results still appear below."));
    return;
  }
  const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([1.2855, 103.8575], 15);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  mapState.base = L.layerGroup().addTo(map);
  mapState.overlay = L.layerGroup().addTo(map);
  map.on("zoomend", updateCorridorWidth);
  mapState.map = map;
}

function corridorPixels() {
  const { map } = mapState;
  const lat = map.getCenter().lat;
  const metresPerPixel = (40075016.686 * Math.cos((lat * Math.PI) / 180)) / 2 ** (map.getZoom() + 8);
  return Math.max(3, (2 * state.delta) / metresPerPixel);
}

function updateCorridorWidth() {
  if (mapState.corridor) mapState.corridor.setStyle({ weight: corridorPixels() });
}

const latlngs = (points) => points.map((p) => (Array.isArray(p) ? [p[0], p[1]] : [p.lat, p.lon]));

function drawInputs() {
  const { map } = mapState;
  if (!map || !state.course || !state.track) return;
  mapState.base.clearLayers();
  mapState.overlay.clearLayers();
  mapState.glitchMarkers.clear();
  const course = latlngs(state.course);
  const track = latlngs(state.track);
  mapState.corridor = L.polyline(course, {
    color: cssVar("--course"),
    opacity: 0.17,
    weight: corridorPixels(),
    lineCap: "round",
    lineJoin: "round",
    interactive: false,
  }).addTo(mapState.base);
  L.polyline(course, { color: cssVar("--course"), weight: 2.5, opacity: 0.95, interactive: false }).addTo(mapState.base);
  L.polyline(track, { color: cssVar("--track"), weight: 1.6, opacity: 0.8, interactive: false }).addTo(mapState.base);
  L.circleMarker(course[0], {
    radius: 6,
    color: cssVar("--ink"),
    weight: 2,
    fillColor: cssVar("--card"),
    fillOpacity: 1,
  })
    .bindTooltip(tip("Start"))
    .addTo(mapState.base);
  L.circleMarker(course[course.length - 1], {
    radius: 3,
    color: cssVar("--ink"),
    weight: 2,
    fillColor: cssVar("--ink"),
    fillOpacity: 1,
  })
    .bindTooltip(tip("Finish"))
    .addTo(mapState.base);
  map.fitBounds(L.latLngBounds([...course, ...track]).pad(0.06));
  updateCorridorWidth();
}

function drawResult(result) {
  const { map } = mapState;
  if (!map) return;
  mapState.overlay.clearLayers();
  mapState.glitchMarkers.clear();
  const offCourse = result.verdict.code === "off_course";

  if (result.cleaned_track) {
    L.polyline(latlngs(result.cleaned_track), {
      color: cssVar("--clean"),
      weight: 2.4,
      opacity: 0.95,
      interactive: false,
    }).addTo(mapState.overlay);
  }
  if (offCourse) {
    const offSet = new Set(result.off_corridor);
    for (const index of offSet) {
      const [lat, lon] = result.track[index];
      L.circleMarker([lat, lon], { radius: 2.5, stroke: false, fillColor: cssVar("--off"), fillOpacity: 0.85 })
        .addTo(mapState.overlay);
    }
    for (const [lat, lon] of result.uncovered_course) {
      L.circleMarker([lat, lon], { radius: 9, color: cssVar("--off"), weight: 2, fill: false, dashArray: "3 3" })
        .bindTooltip(tip("Course point never approached"))
        .addTo(mapState.overlay);
    }
  }
  for (const glitch of result.glitches) {
    const marker = L.circleMarker([glitch.lat, glitch.lon], {
      radius: 6.5,
      color: cssVar("--card"),
      weight: 2,
      fillColor: cssVar("--glitch"),
      fillOpacity: 1,
    })
      .bindTooltip(tip(`Fix #${glitch.index} · ${glitch.offset_m} m off course`))
      .addTo(mapState.overlay);
    mapState.glitchMarkers.set(glitch.index, marker);
  }
  mapState.hover = L.circleMarker([0, 0], {
    radius: 7,
    color: cssVar("--ink"),
    weight: 2,
    fill: false,
    interactive: false,
    opacity: 0,
  }).addTo(mapState.overlay);
}

function focusFix(index) {
  const { map } = mapState;
  if (!map || !state.result) return;
  const [lat, lon] = state.result.track[index];
  map.flyTo([lat, lon], Math.max(map.getZoom(), 17), { duration: 0.6 });
  const marker = mapState.glitchMarkers.get(index);
  if (marker) marker.openTooltip();
}

// ------------------------------------------------------------------ trace

const trace = { positions: null, scale: null };

function renderTrace(result, animate = true) {
  const svgEl = $("#trace");
  svgEl.replaceChildren();
  const width = Math.max(320, svgEl.clientWidth || 600);
  const height = 150;
  svgEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const pad = { left: 40, right: 10, top: 10, bottom: 20 };
  const offsets = result.offsets_m;
  const n = offsets.length;
  const delta = result.delta_m;
  const yMax = Math.max(3 * delta, ...offsets);
  const x = (pos) => pad.left + (n <= 1 ? 0 : (pos / (n - 1)) * (width - pad.left - pad.right));
  const y = (value) => pad.top + (1 - Math.sqrt(Math.max(0, value) / yMax)) * (height - pad.top - pad.bottom);
  const base = height - pad.bottom;
  trace.scale = { x, y, n, pad, width };

  const clipId = "above-corridor";
  const defs = s("defs");
  const clip = s("clipPath", { id: clipId });
  clip.append(s("rect", { x: 0, y: 0, width, height: y(delta) }));
  defs.append(clip);
  svgEl.append(defs);

  svgEl.append(s("rect", { class: "trace-band", x: pad.left, y: y(delta), width: width - pad.left - pad.right, height: base - y(delta) }));
  for (const tick of [0, delta, yMax]) {
    svgEl.append(s("line", { class: "trace-grid", x1: pad.left, x2: width - pad.right, y1: y(tick), y2: y(tick) }));
    const label = s("text", { class: "trace-axis", x: pad.left - 6, y: y(tick) + 3, "text-anchor": "end" });
    label.textContent = tick === delta ? `δ ${Math.round(delta)}` : `${Math.round(tick)} m`;
    svgEl.append(label);
  }
  svgEl.append(s("line", { class: "trace-delta", x1: pad.left, x2: width - pad.right, y1: y(delta), y2: y(delta) }));

  const d = offsets.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const line = s("path", { class: "trace-line", d });
  const excess = s("path", { class: "trace-line", d, "clip-path": `url(#${clipId})` });
  // Inline style via the CSSOM outranks the class rule and is allowed by the CSP.
  excess.style.stroke = result.verdict.code === "off_course" ? cssVar("--off") : cssVar("--glitch");
  excess.style.strokeWidth = "1.8";
  svgEl.append(line, excess);
  if (animate && line.getTotalLength) {
    const length = String(line.getTotalLength());
    for (const path of [line, excess]) {
      path.style.setProperty("--len", length);
      path.style.strokeDasharray = length;
      path.classList.add("is-drawn");
    }
  }

  const positionOf = new Map(result.analysed_indices.map((idx, pos) => [idx, pos]));
  trace.positions = result.analysed_indices;
  for (const glitch of result.glitches) {
    const pos = positionOf.get(glitch.index);
    if (pos === undefined) continue;
    svgEl.append(s("line", { class: "trace-spike", x1: x(pos), x2: x(pos), y1: y(delta), y2: y(offsets[pos]) }));
    svgEl.append(s("circle", { class: "trace-glitch", cx: x(pos), cy: y(offsets[pos]), r: 4 }));
  }
  const cursor = s("line", { class: "trace-cursor", x1: 0, x2: 0, y1: pad.top, y2: base, opacity: 0 });
  svgEl.append(cursor);
  trace.cursor = cursor;

  const aboveCount = offsets.filter((v) => v > delta).length;
  svgEl.setAttribute(
    "aria-label",
    `Distance from the course for ${n} analysed fixes. ${aboveCount} are more than ${delta} metres away; ` +
      `${result.glitches.length} were discarded as glitches. Use the left and right arrow keys to step through fixes.`,
  );
  svgEl.setAttribute("tabindex", "0");
}

function traceAt(pos) {
  const { result } = state;
  if (!result || !trace.scale) return;
  const { x, n } = trace.scale;
  const clamped = Math.min(n - 1, Math.max(0, pos));
  trace.current = clamped;
  const index = trace.positions[clamped];
  const offset = result.offsets_m[clamped];
  trace.cursor.setAttribute("x1", x(clamped));
  trace.cursor.setAttribute("x2", x(clamped));
  trace.cursor.setAttribute("opacity", 1);
  const time = fmtTime(state.track[index]?.time);
  $("#trace-readout").textContent = `fix #${index}${time ? " · " + time : ""} · ${offset.toFixed(1)} m`;
  if (mapState.hover) {
    const [lat, lon] = result.track[index];
    mapState.hover.setLatLng([lat, lon]).setStyle({ opacity: 1 });
  }
}

function traceClear() {
  if (trace.cursor) trace.cursor.setAttribute("opacity", 0);
  if (mapState.hover) mapState.hover.setStyle({ opacity: 0 });
  $("#trace-readout").textContent = "";
}

function initTraceInteraction() {
  const svgEl = $("#trace");
  svgEl.addEventListener("pointermove", (event) => {
    if (!trace.scale) return;
    const rect = svgEl.getBoundingClientRect();
    const { pad, width, n } = trace.scale;
    const px = ((event.clientX - rect.left) / rect.width) * width;
    const pos = Math.round(((px - pad.left) / (width - pad.left - pad.right)) * (n - 1));
    traceAt(pos);
  });
  svgEl.addEventListener("pointerleave", traceClear);
  svgEl.addEventListener("keydown", (event) => {
    if (!trace.scale) return;
    const step = event.shiftKey ? 10 : 1;
    if (event.key === "ArrowRight") traceAt((trace.current ?? -1) + step);
    else if (event.key === "ArrowLeft") traceAt((trace.current ?? 1) - step);
    else return;
    event.preventDefault();
  });
  svgEl.addEventListener("blur", traceClear);
}

// ------------------------------------------------------------------ report

function fmtTime(iso) {
  if (!iso) return "";
  const match = /T(\d{2}:\d{2}:\d{2})/.exec(iso);
  return match ? match[1] : iso.slice(0, 19);
}

const mark = (passed) =>
  passed === null || passed === undefined
    ? h("span", { class: "mark-na" }, "—")
    : h("span", { class: passed ? "mark-pass" : "mark-fail" }, passed ? "✓ pass" : "✗ fail");

function countCell(value, status, budget) {
  if (value !== null && value !== undefined) return `${value} ${value === 1 ? "fix" : "fixes"}`;
  if (status === "budget_exceeded") return `> ${budget} fixes`;
  return "impossible";
}

function renderReport(result) {
  const report = $("#report");
  const { verdict, checks, processing, verification } = result;
  const [main, ...rest] = verdict.headline.split(" · ");
  const stamp = h(
    "p",
    { class: `stamp stamp-${verdict.code} is-landing` },
    main,
    rest.length ? h("span", { class: "stamp-sub" }, rest.join(" · ")) : null,
  );

  const figures = h(
    "dl",
    { class: "figures" },
    h("div", { class: "figure" }, h("dt", {}, "Glitches removed"), h("dd", {}, checks.continuous_edit_distance ?? "—")),
    h(
      "div",
      { class: "figure" },
      h("dt", {}, "Fixes analysed"),
      h("dd", {}, processing.track_points_analysed, h("small", {}, `of ${processing.track_points}`)),
    ),
    h("div", { class: "figure" }, h("dt", {}, "Largest offset"), h("dd", {}, checks.max_offset_m, h("small", {}, "m"))),
    h("div", { class: "figure" }, h("dt", {}, "Solved in"), h("dd", {}, Math.round(processing.solver_ms), h("small", {}, "ms"))),
  );

  const row = (title, note, cell, engine = false) =>
    h("tr", { class: engine ? "is-engine" : null }, h("th", { scope: "row" }, title, h("small", {}, note)), h("td", {}, cell));
  const compare = h(
    "table",
    { class: "compare" },
    h(
      "tbody",
      {},
      row("Every fix within ±δ of the course", `order-blind; one glitch fails it (max ${checks.max_offset_m} m)`, mark(checks.fixes_near_course)),
      row("Every course point visited", `order-blind (farthest ${checks.max_uncovered_m} m)`, mark(checks.course_covered)),
      row("Fréchet distance ≤ δ, as recorded", "order-aware; one glitch still fails it", mark(checks.ordinary_frechet_within)),
      row(
        "Discrete Fréchet edit distance",
        "matches fixes to course vertices only",
        countCell(checks.discrete_edit_distance, checks.discrete_status, null),
      ),
      row(
        "Continuous Fréchet edit distance",
        "this audit: order-aware and glitch-tolerant",
        countCell(checks.continuous_edit_distance, checks.continuous_status, checks.glitch_budget),
        true,
      ),
    ),
  );

  const parts = [
    h("p", { class: "kicker" }, "Verdict"),
    stamp,
    h("p", { class: "verdict-detail" }, verdict.detail),
    figures,
    h("h3", {}, "Why the simpler checks are not enough"),
    compare,
  ];

  if (result.glitches.length) {
    const rows = result.glitches.map((glitch) =>
      h(
        "tr",
        {},
        h(
          "td",
          { colspan: 3 },
          h(
            "button",
            { type: "button", onclick: () => focusFix(glitch.index), "aria-label": `Show fix ${glitch.index} on the map` },
            h("span", { class: "idx" }, `#${glitch.index}`),
            h("span", {}, fmtTime(glitch.time) || "—"),
            h("span", { class: "num" }, `${glitch.offset_m} m`),
          ),
        ),
      ),
    );
    parts.push(
      h("h3", {}, "Discarded fixes", h("span", { class: "count" }, `${result.glitches.length}`)),
      h(
        "table",
        { class: "ledger" },
        h("thead", {}, h("tr", {}, h("th", {}, "fix"), h("th", {}, "time"), h("th", { class: "num" }, "offset"))),
        h("tbody", {}, rows),
      ),
    );
  }

  const actions = h("div", { class: "actions" });
  if (result.cleaned_track && result.glitches.length) {
    actions.append(h("button", { type: "button", class: "ghost", onclick: downloadCleaned }, "Cleaned track · GPX"));
  }
  actions.append(h("button", { type: "button", class: "ghost", onclick: downloadReport }, "Full report · JSON"));
  parts.push(actions);

  const sealText = {
    verified: "✓ Independently verified",
    skipped: "○ Verification skipped",
    failed: "✗ Verification FAILED",
    not_applicable: "○ Nothing to verify",
  }[verification.status];
  const notes = [
    h("p", {}, h("span", { class: `seal seal-${verification.status}` }, sealText), " — ", verification.detail),
    h(
      "p",
      {},
      `Analysed ${processing.track_points_analysed} of ${processing.track_points} fixes` +
        (processing.track_decimation > 1 ? ` (every ${processing.track_decimation}th, to bound run time)` : "") +
        `; course ${processing.course_points_analysed} of ${processing.course_points} points` +
        (processing.course_simplified_m > 0 ? ` (simplified by ≤ ${processing.course_simplified_m} m)` : "") +
        ".",
    ),
    h(
      "p",
      {},
      `Budgets tried: ${processing.budgets_tried.join(", ")}. Glitch budget ${checks.glitch_budget}. ` +
        `Local projection scale error ≤ ${(processing.projection_scale_error * 100).toFixed(4)}%.`,
    ),
  ];
  parts.push(h("div", { class: "provenance" }, notes));

  report.replaceChildren(...parts);
}

function renderEmpty(message) {
  $("#report").replaceChildren(h("div", { class: "report-empty" }, h("p", { class: "kicker" }, "Ready"), h("p", {}, message)));
  $("#trace").replaceChildren();
  $("#trace").setAttribute("aria-label", "No audit yet");
  trace.scale = null;
}

// ------------------------------------------------------------------ downloads

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function downloadCleaned() {
  if (!state.result?.cleaned_track) return;
  try {
    const response = await fetch("/api/gpx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: "cleaned-track",
        points: state.result.cleaned_track.map(([lat, lon, time]) => ({ lat, lon, time })),
      }),
    });
    if (!response.ok) throw new Error(`export failed (${response.status})`);
    saveBlob(await response.blob(), "cleaned-track.gpx");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function downloadReport() {
  if (!state.result) return;
  saveBlob(new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" }), "course-audit.json");
}

// ------------------------------------------------------------------ inputs

function refreshRunButton() {
  $("#run").disabled = state.busy || !(state.course && state.track);
}

function setDelta(value) {
  state.delta = Number(value);
  $("#delta").value = String(state.delta);
  $("#delta-out").textContent = `±${state.delta} m`;
  updateCorridorWidth();
}

async function loadScenarios() {
  const list = $("#scenario-list");
  try {
    const items = await api("/api/scenarios");
    items.forEach((item, i) => {
      const button = h(
        "button",
        { type: "button", class: "scenario", "aria-pressed": "false", "data-id": item.id, onclick: () => pickScenario(item.id) },
        h("span", { class: "scenario-letter" }, String.fromCharCode(65 + i)),
        h("span", { class: "scenario-title" }, item.title),
        h("span", { class: "scenario-expect" }, item.expectation),
      );
      list.append(h("li", {}, button));
    });
  } catch (error) {
    setStatus(`Could not load scenarios: ${error.message}`, true);
  }
}

async function pickScenario(id) {
  setStatus("Loading scenario…");
  try {
    const scenario = await api(`/api/scenarios/${encodeURIComponent(id)}`);
    state.scenarioId = id;
    state.course = scenario.course;
    state.track = scenario.track;
    state.courseLabel = `${scenario.title} (course)`;
    state.trackLabel = `${scenario.title} (track)`;
    state.result = null;
    for (const button of document.querySelectorAll(".scenario")) {
      button.setAttribute("aria-pressed", String(button.dataset.id === id));
    }
    resetDrop("course");
    resetDrop("track");
    setDelta(scenario.delta_m);
    drawInputs();
    renderEmpty(`${scenario.blurb} Press “Run audit”.`);
    setStatus(`${scenario.course.length} course points · ${scenario.track.length} fixes`);
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    refreshRunButton();
  }
}

function resetDrop(role) {
  const drop = $(`#${role}-drop`);
  drop.classList.remove("is-loaded", "is-error");
  $(`#${role}-status`).textContent = "no file";
  $(`#${role}-file`).value = "";
}

async function uploadFile(role, file) {
  const drop = $(`#${role}-drop`);
  const status = $(`#${role}-status`);
  drop.classList.remove("is-loaded", "is-error");
  status.textContent = "reading…";
  const form = new FormData();
  form.append("file", file);
  try {
    const data = await api("/api/parse", { method: "POST", body: form });
    state[role] = data.points;
    state[`${role}Label`] = file.name;
    state.scenarioId = null;
    state.result = null;
    for (const button of document.querySelectorAll(".scenario")) button.setAttribute("aria-pressed", "false");
    drop.classList.add("is-loaded");
    status.textContent = `${file.name} · ${data.points.length} points`;
    drawInputs();
    renderEmpty(state.course && state.track ? "Both files loaded. Press “Run audit”." : `Now add a ${role === "course" ? "track" : "course"}.`);
    setStatus("");
  } catch (error) {
    drop.classList.add("is-error");
    status.textContent = error.message;
  } finally {
    refreshRunButton();
  }
}

function initUploads() {
  for (const role of ["course", "track"]) {
    const input = $(`#${role}-file`);
    const drop = $(`#${role}-drop`);
    input.addEventListener("change", () => {
      if (input.files?.[0]) uploadFile(role, input.files[0]);
    });
    drop.addEventListener("dragover", (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
    drop.addEventListener("dragleave", () => drop.classList.remove("is-over"));
    drop.addEventListener("drop", (event) => {
      event.preventDefault();
      drop.classList.remove("is-over");
      const file = event.dataTransfer?.files?.[0];
      if (file) uploadFile(role, file);
    });
  }
}

async function runAudit() {
  if (!state.course || !state.track || state.busy) return;
  state.busy = true;
  refreshRunButton();
  const button = $("#run");
  button.classList.add("is-busy");
  $(".run-label", button).textContent = "Tracing free space";
  setStatus("Searching for the fewest fixes to discard…");
  try {
    const result = await postJson("/api/audit", {
      course: state.course,
      track: state.track,
      delta_m: state.delta,
    });
    state.result = result;
    drawResult(result);
    renderReport(result);
    renderTrace(result);
    setStatus(`Done in ${Math.round(result.processing.total_ms)} ms.`);
    $("#report").focus({ preventScroll: true });
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    state.busy = false;
    button.classList.remove("is-busy");
    $(".run-label", button).textContent = "Run audit";
    refreshRunButton();
  }
}

function init() {
  initMap();
  initUploads();
  initTraceInteraction();
  $("#delta").addEventListener("input", (event) => {
    setDelta(event.target.value);
    if (state.result) setStatus("Tolerance changed - run the audit again.");
  });
  $("#run").addEventListener("click", runAudit);
  let resizeTimer = 0;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.result) renderTrace(state.result, false);
    }, 150);
  });
  loadScenarios();
}

init();

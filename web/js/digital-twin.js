// ============================================================================
// Space Traffic Digital Twin — browser twin
//
// This file is the canonical browser implementation of the STM digital twin
// and is mirrored byte-for-byte in two repositories:
//   · PHG04/space_traffic_digital_twin  → web/js/digital-twin.js   (home)
//   · PHG04/website                     → assets/js/digital-twin.js (mirror)
// Keep them identical — edit once, copy to the other.
//
// It follows the Python package layout of the research sim; each numbered
// section notes its counterpart:
//   §2  data management        ↔ src/data_management/satellite_manager.py
//   §3  orbital mechanics      ↔ src/orbital_mechanics/{propagator,orbit_engine}.py
//   §4  spatial indexing       ↔ src/conjunction_detection/spatial_index.py
//   §5  conjunction detection  ↔ src/conjunction_detection/conjunction_analyzer.py
//   §6  risk assessment        ↔ src/risk_assessment/collision_probability.py
//   §7  sensor simulation      ↔ src/sensor_simulation/noise_models.py
//   §8  manoeuvre planning     (twin-side what-if; Clohessy–Wiltshire)
//   §9  prediction scan        (the twin running ahead of the wall clock)
//   §10+ dashboard             ↔ dashboard/app.py (Dash → Three.js)
//
// The demo teaches the digital-twin idea in four beats:
//   MIRROR  — a live copy of the real sky (CelesTrak GP data, SGP4)
//   PREDICT — the copy runs faster than reality and screens the future
//   ACT     — try a collision-avoidance burn on the copy, never the sky
//   EXPLORE — free sandbox incl. the noisy-sensor research mode
// ============================================================================
import * as THREE from "three";
import * as satellite from "satellite";

// ============================================================================
// §1 · Constants + small utilities
// ============================================================================
const MU       = 398600.4418;     // km³/s² (GM earth)
const R_EARTH  = 6378.137;        // km
const J2       = 1.08262668e-3;
const DEG      = Math.PI / 180;
const TWO_PI   = Math.PI * 2;
const SCENE_SCALE = 1 / 1000;     // 1 scene unit = 1000 km
const VREL_MAX = 16;              // km/s — max plausible LEO relative speed

const COL = {
  mint:     0x8fd5c0,
  mintHi:   0xb6e5d6,
  coral:    0xff8c6a,
  ink:      0xe8edf5,
  blue:     0x8ac5e0,
  lavender: 0xc9a8d8,
};

const clamp  = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp   = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussOf(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(TWO_PI * v);
}

const pad2 = n => (n < 10 ? "0" : "") + n;
function fmtUTC(ms, withSeconds = true) {
  const d = new Date(ms);
  const hm = `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`;
  return withSeconds ? `${hm}:${pad2(d.getUTCSeconds())}` : hm;
}
function fmtOffset(ms) {
  const sign = ms < 0 ? "−" : "+";
  let s = Math.round(Math.abs(ms) / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); s = s % 60;
  if (h > 0) return `${sign}${h}h ${pad2(m)}m`;
  if (m > 0) return `${sign}${m}m ${pad2(s)}s`;
  return `${sign}${s}s`;
}
function fmtCountdown(ms) {
  if (ms <= 0) return "passed";
  let s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); s = s % 60;
  if (h > 0) return `in ${h}h ${pad2(m)}m`;
  if (m > 0) return `in ${m}m ${pad2(s)}s`;
  return `in ${s}s`;
}
function fmtDist(km) {
  if (!isFinite(km)) return "—";
  if (km < 1) return `${Math.round(km * 1000)} m`;
  return `${km.toFixed(km < 10 ? 2 : 1)} km`;
}
function fmtAge(ms) {
  const m = Math.round(ms / 60000);
  if (m < 1)  return "under a minute old";
  if (m < 90) return `${m} min old`;
  const h = Math.round(m / 6) / 10;
  if (h < 48) return `${h} h old`;
  return `${Math.round(h / 24)} days old`;
}
function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function cleanName(raw, idx) {
  if (!raw) return `OBJ-${String(idx).padStart(4, "0")}`;
  return raw.replace(/^0 /, "").trim();
}

// ============================================================================
// §2 · Data management — catalog loading
// ↔ src/data_management/satellite_manager.py
//
// Live path: same-origin proxy (/api/tle) → CelesTrak GP data, CDN-cached 2 h.
// Fallback: TLE snapshot bundled next to this file (manifest carries its date).
// ============================================================================
const SOURCES = {
  active:   { label: "Active LEO catalog", file: "active-leo.tle" },
  starlink: { label: "Starlink",           file: "starlink.tle"   },
  oneweb:   { label: "OneWeb",             file: "oneweb.tle"     },
};
const FAMOUS = {
  25544: "ISS",
  48274: "Tiangong",
  20580: "Hubble",
};

function parseTLE(text) {
  const lines = text.split(/\r?\n/);
  const out = [];
  for (let i = 0; i + 2 < lines.length; i++) {
    const name = lines[i], l1 = lines[i + 1], l2 = lines[i + 2];
    if (!l1 || !l2 || !l1.startsWith("1 ") || !l2.startsWith("2 ")) continue;
    let satrec;
    try { satrec = satellite.twoline2satrec(l1, l2); } catch (_) { continue; }
    if (!satrec || satrec.error) continue;
    const revPerDay = (satrec.no || 0) * 1440 / TWO_PI;
    if (revPerDay < 11 || revPerDay > 17) continue;   // LEO only
    out.push({ name: cleanName(name, out.length), satrec, satnum: +satrec.satnum });
    i += 2;
  }
  return out;
}

async function fetchText(url, timeoutMs) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctl.signal });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text = await r.text();
    return { text, fetchedAt: r.headers.get("X-Fetched-At") };
  } finally {
    clearTimeout(timer);
  }
}

// Loads one source group + the stations group (so the ISS is always aboard),
// preferring the live proxy and falling back to the bundled snapshot.
async function loadCatalog(sourceKey, { apiBase, dataBase }) {
  const src = SOURCES[sourceKey] || SOURCES.active;
  const tryPaths = [];
  if (apiBase != null) {
    tryPaths.push({
      live: true,
      group: `${apiBase}/tle?group=${sourceKey}`,
      stations: `${apiBase}/tle?group=stations`,
    });
  }
  tryPaths.push({
    live: false,
    group: `${dataBase}${src.file}`,
    stations: `${dataBase}stations.tle`,
  });

  for (const p of tryPaths) {
    try {
      const [g, st] = await Promise.all([
        fetchText(p.group, p.live ? 15000 : 20000),
        fetchText(p.stations, p.live ? 15000 : 20000).catch(() => null),
      ]);
      const main = parseTLE(g.text);
      if (main.length < 10) throw new Error("catalog too small — bad payload?");
      const stations = st ? parseTLE(st.text) : [];
      let fetchedAt = g.fetchedAt ? Date.parse(g.fetchedAt) : NaN;
      if (!p.live && !isFinite(fetchedAt)) {
        try {
          const mf = await (await fetch(`${dataBase}tle-manifest.json`)).json();
          fetchedAt = Date.parse(mf.fetched_at);
        } catch (_) { /* manifest optional */ }
      }
      return { main, stations, live: p.live, fetchedAt: isFinite(fetchedAt) ? fetchedAt : null };
    } catch (e) {
      console.warn(`[digital-twin] catalog path failed (${p.live ? "live" : "snapshot"}):`, e.message || e);
    }
  }
  return null;
}

// Even-stride subset of the catalog (preserves orbital-plane diversity),
// with famous spacecraft force-included. Only the famous entries are taken
// from the stations group — it also lists ISS modules and docked vehicles as
// separate objects at the same position, which would duplicate every
// ISS conjunction several times over.
function buildFleet(catalog, n) {
  const byNum = new Map();
  for (const rec of catalog.stations) if (FAMOUS[rec.satnum]) byNum.set(rec.satnum, rec);
  for (const rec of catalog.main) if (!byNum.has(rec.satnum)) byNum.set(rec.satnum, rec);
  const all = Array.from(byNum.values());
  const famous = all.filter(r => FAMOUS[r.satnum]);
  const rest   = all.filter(r => !FAMOUS[r.satnum]);
  const want   = Math.min(n, all.length);
  const picked = [...famous];
  const step   = rest.length / Math.max(want - famous.length, 1);
  for (let k = 0; picked.length < want && k * step < rest.length; k++) {
    picked.push(rest[Math.floor(k * step)]);
  }
  return picked.map((rec, id) => ({
    id,
    name: rec.name,
    satnum: rec.satnum,
    famous: FAMOUS[rec.satnum] || null,
    real: true,
    satrec: rec.satrec,
    epochMs: (rec.satrec.jdsatepoch - 2440587.5) * 86400000,
  }));
}

// ============================================================================
// §3 · Orbital mechanics
// ↔ src/orbital_mechanics/propagator.py (KeplerianPropagator / J2Propagator)
// ↔ src/orbital_mechanics/orbit_engine.py (STMOrbitEngine.propagate_all)
//
// Real objects: SGP4 (satellite.js) straight from the TLE, called with minutes
// since the TLE's own epoch — no Date allocation in the hot path.
// Synthetic fallback fleet: Kepler + J2 secular rates, the same model as the
// Python J2Propagator.
// ============================================================================
function generateSyntheticFleet(n, seed = 1) {
  const rng = mulberry32(seed);
  const shells = [
    { alt: 550,  inc: 53.0, w: 0.35 }, { alt: 540,  inc: 70.0, w: 0.10 },
    { alt: 1200, inc: 87.0, w: 0.15 }, { alt: 780,  inc: 86.4, w: 0.10 },
    { alt: 408,  inc: 51.6, w: 0.05 }, { alt: 700,  inc: 98.2, w: 0.15 },
    { alt: 1100, inc: 53.0, w: 0.10 },
  ];
  const sats = new Array(n);
  for (let k = 0; k < n; k++) {
    const r = rng();
    let cum = 0, shell = shells[0];
    for (const s of shells) { cum += s.w; if (r <= cum) { shell = s; break; } }
    const a = R_EARTH + shell.alt + (rng() - 0.5) * 60;
    const e = rng() * 0.004;
    const i = (shell.inc + (rng() - 0.5) * 4) * DEG;
    const nMean = Math.sqrt(MU / (a * a * a));
    const p = a * (1 - e * e);
    const fac = 1.5 * J2 * (R_EARTH / p) * (R_EARTH / p);
    const cosi = Math.cos(i), sini = Math.sin(i);
    sats[k] = {
      id: k, name: `SYN-${String(k).padStart(4, "0")}`, real: false, famous: null,
      a, e, sini, cosi,
      raan0: rng() * TWO_PI, argp0: rng() * TWO_PI, m0: rng() * TWO_PI,
      raanDot: -fac * nMean * cosi,
      argpDot:  fac * nMean * 0.5 * (5 * cosi * cosi - 1),
      mDot:     nMean * (1 + fac * Math.sqrt(1 - e * e) * (1 - 1.5 * sini * sini)),
    };
  }
  return sats;
}

const FAR = 1e7; // parking spot for decayed/failed propagations (km)

function propagateKepler(sat, tSec, out) {
  const raan = sat.raan0 + sat.raanDot * tSec;
  const argp = sat.argp0 + sat.argpDot * tSec;
  const M    = sat.m0    + sat.mDot    * tSec;
  const e    = sat.e;
  let E = M;
  for (let k = 0; k < 5; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  const cosE = Math.cos(E), sinE = Math.sin(E);
  const r = sat.a * (1 - e * cosE);
  const cosNu = (cosE - e) / (1 - e * cosE);
  const sinNu = (Math.sqrt(1 - e * e) * sinE) / (1 - e * cosE);
  const xp = r * cosNu, yp = r * sinNu;
  const cosO = Math.cos(raan), sinO = Math.sin(raan);
  const cosw = Math.cos(argp), sinw = Math.sin(argp);
  const ci = sat.cosi, si = sat.sini;
  out[0] = (cosO * cosw - sinO * sinw * ci) * xp + (-cosO * sinw - sinO * cosw * ci) * yp;
  out[1] = (sinO * cosw + cosO * sinw * ci) * xp + (-sinO * sinw + cosO * cosw * ci) * yp;
  out[2] = (sinw * si) * xp + (cosw * si) * yp;
}

// Position+velocity at absolute time tMs → ECI km, km/s.
function propagateECI(sat, tMs, pos, vel) {
  if (sat.real) {
    const pv = satellite.sgp4(sat.satrec, (tMs - sat.epochMs) / 60000);
    if (!pv || !pv.position || !isFinite(pv.position.x)) {
      pos[0] = FAR; pos[1] = FAR; pos[2] = FAR;
      if (vel) { vel[0] = 0; vel[1] = 0; vel[2] = 0; }
      return false;
    }
    pos[0] = pv.position.x; pos[1] = pv.position.y; pos[2] = pv.position.z;
    if (vel) { vel[0] = pv.velocity.x; vel[1] = pv.velocity.y; vel[2] = pv.velocity.z; }
    return true;
  }
  const tSec = tMs / 1000;
  propagateKepler(sat, tSec, pos);
  if (vel) {
    const p2 = _kepTmp;
    propagateKepler(sat, tSec + 0.5, p2);
    vel[0] = (p2[0] - pos[0]) * 2; vel[1] = (p2[1] - pos[1]) * 2; vel[2] = (p2[2] - pos[2]) * 2;
  }
  return true;
}
const _kepTmp = [0, 0, 0];

function orbitalPeriodMs(sat) {
  if (sat.real) return (TWO_PI / sat.satrec.no) * 60000;   // satrec.no in rad/min
  return TWO_PI / Math.sqrt(MU / (sat.a ** 3)) * 1000;
}

function gmstAt(tMs) {
  return satellite.gstime(new Date(tMs));
}

// --- Propagation window: the engine behind each rendered globe. -------------
// Positions are computed at fixed sim substeps [tA, tB]; rendering linearly
// interpolates between them, so the hot path stays cheap at 1× and detection
// always works on consistent substep state (↔ real_time_engine.py StateCache).
class PropagationWindow {
  constructor(fleet) {
    this.fleet = fleet;
    const n = fleet.length;
    this.n = n;
    this.posA = new Float64Array(n * 3); this.posB = new Float64Array(n * 3);
    this.velA = new Float64Array(n * 3); this.velB = new Float64Array(n * 3);
    this.tA = NaN; this.tB = NaN;
    this.postHooks = new Map();          // satIdx → fn(tMs, pos, vel) (manoeuvre overlay)
    this.onSubstep = null;               // fn(tA, tB, posA, velA, dtSec) — screening hook
  }
  setPostHook(idx, fn) { if (fn) this.postHooks.set(idx, fn); else this.postHooks.delete(idx); }
  fill(tMs, pos, vel) {
    const f = this.fleet, p = [0, 0, 0], v = [0, 0, 0];
    for (let i = 0; i < this.n; i++) {
      propagateECI(f[i], tMs, p, v);
      const hook = this.postHooks.get(i);
      if (hook) hook(tMs, p, v);
      pos[i * 3] = p[0]; pos[i * 3 + 1] = p[1]; pos[i * 3 + 2] = p[2];
      vel[i * 3] = v[0]; vel[i * 3 + 1] = v[1]; vel[i * 3 + 2] = v[2];
    }
  }
  // Make [tA,tB] cover tMs with substep dtSec; returns true when a new substep
  // boundary was crossed (callers run screening then).
  ensure(tMs, dtSec) {
    const dtMs = dtSec * 1000;
    if (!isFinite(this.tA) || tMs < this.tA - dtMs || tMs > this.tB + dtMs * 4) {
      // Cold start or a jump (scrub) — recompute both bounds.
      this.tA = tMs; this.tB = tMs + dtMs;
      this.fill(this.tA, this.posA, this.velA);
      this.fill(this.tB, this.posB, this.velB);
      if (this.onSubstep) this.onSubstep(this.tA, this.tB, this.posA, this.velA, dtSec);
      return true;
    }
    let advanced = false;
    while (tMs > this.tB) {
      // Slide the window: B becomes A, propagate a fresh B.
      [this.posA, this.posB] = [this.posB, this.posA];
      [this.velA, this.velB] = [this.velB, this.velA];
      this.tA = this.tB; this.tB = this.tA + dtMs;
      this.fill(this.tB, this.posB, this.velB);
      if (this.onSubstep) this.onSubstep(this.tA, this.tB, this.posA, this.velA, (this.tB - this.tA) / 1000);
      advanced = true;
    }
    return advanced;
  }
  // Interpolated positions at tMs → out (Float64Array n*3).
  sample(tMs, out) {
    const span = this.tB - this.tA;
    const t = span > 0 ? clamp((tMs - this.tA) / span, 0, 1) : 0;
    const a = this.posA, b = this.posB;
    for (let i = 0, m = this.n * 3; i < m; i++) out[i] = a[i] + (b[i] - a[i]) * t;
  }
}

// ============================================================================
// §4 · Spatial indexing — uniform hash grid over ECI space
// ↔ src/conjunction_detection/spatial_index.py (KDTreeSpatialIndex
//   .range_query_pairs). A rebuilt-per-tick KD tree and a uniform grid answer
//   the same query; the grid is the cheaper fit for JS typed arrays.
// ============================================================================
function candidatePairs(pos, n, radiusKm) {
  const inv = 1 / radiusKm;
  const cells = new Map();
  const key = (x, y, z) => x * 73856093 ^ y * 19349663 ^ z * 83492791;
  const ix = new Int32Array(n), iy = new Int32Array(n), iz = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3];
    if (x > 5e6) continue;                       // parked/dead object
    ix[i] = Math.floor(x * inv); iy[i] = Math.floor(pos[i * 3 + 1] * inv); iz[i] = Math.floor(pos[i * 3 + 2] * inv);
    const k = key(ix[i], iy[i], iz[i]);
    let b = cells.get(k); if (!b) { b = []; cells.set(k, b); } b.push(i);
  }
  const out = [];
  const r2 = radiusKm * radiusKm;
  for (let i = 0; i < n; i++) {
    if (pos[i * 3] > 5e6) continue;
    const xi = pos[i * 3], yi = pos[i * 3 + 1], zi = pos[i * 3 + 2];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const bkt = cells.get(key(ix[i] + dx, iy[i] + dy, iz[i] + dz));
      if (!bkt) continue;
      for (const j of bkt) {
        if (j <= i) continue;
        const ddx = pos[j * 3] - xi, ddy = pos[j * 3 + 1] - yi, ddz = pos[j * 3 + 2] - zi;
        if (ddx * ddx + ddy * ddy + ddz * ddz < r2) out.push(i, j);
      }
    }
  }
  return out;
}

// ============================================================================
// §5 · Conjunction detection
// ↔ src/conjunction_detection/conjunction_analyzer.py (ConjunctionAnalyzer)
//
// Between substeps relative motion is ~linear, so each candidate pair gets an
// analytic time-of-closest-approach inside the substep — a close pass cannot
// slip between samples no matter the playback speed (the v1 demo's big flaw).
// ============================================================================
function closestApproach(pos, vel, i, j, dtSec) {
  const rx = pos[j * 3] - pos[i * 3], ry = pos[j * 3 + 1] - pos[i * 3 + 1], rz = pos[j * 3 + 2] - pos[i * 3 + 2];
  const vx = vel[j * 3] - vel[i * 3], vy = vel[j * 3 + 1] - vel[i * 3 + 1], vz = vel[j * 3 + 2] - vel[i * 3 + 2];
  const v2 = vx * vx + vy * vy + vz * vz;
  let t = v2 > 1e-12 ? -(rx * vx + ry * vy + rz * vz) / v2 : 0;
  t = clamp(t, 0, dtSec);
  const cx = rx + vx * t, cy = ry + vy * t, cz = rz + vz * t;
  return { tSec: t, dKm: Math.sqrt(cx * cx + cy * cy + cz * cz), vrel: Math.sqrt(v2) };
}

// Screen one substep: hash radius covers the farthest two objects can travel
// toward each other in dtSec, so the linear refinement sees every candidate.
// The relative-speed floor drops co-moving pairs — docked vehicles (the ISS
// catalog entry is surrounded by berthed spacecraft) and formation flyers are
// not conjunctions.
const VREL_MIN = 0.15; // km/s
function screenSubstep(pos, vel, n, thresholdKm, dtSec) {
  const radius = thresholdKm + VREL_MAX * dtSec;
  const cand = candidatePairs(pos, n, radius);
  const hits = [];
  for (let k = 0; k < cand.length; k += 2) {
    const i = cand[k], j = cand[k + 1];
    const ca = closestApproach(pos, vel, i, j, dtSec);
    if (ca.dKm < thresholdKm && ca.vrel > VREL_MIN) hits.push({ i, j, dKm: ca.dKm, tSec: ca.tSec, vrel: ca.vrel });
  }
  return hits;
}

// Live event tracker: one ongoing close pass = one event, not a stream of
// duplicates (↔ ConjunctionAnalyzer._update_conjunction_event).
class ConjunctionTracker {
  constructor() { this.events = new Map(); this.total = 0; this.lingerMs = 4000; }
  reset() { this.events.clear(); this.total = 0; }
  update(hits, tAMs, nowRealMs) {
    for (const h of hits) {
      const key = `${h.i}-${h.j}`;
      let ev = this.events.get(key);
      if (!ev) {
        ev = { key, a: h.i, b: h.j, minDist: h.dKm, tcaMs: tAMs + h.tSec * 1000, vrel: h.vrel, live: true, lastSeen: nowRealMs };
        this.events.set(key, ev);
        this.total++;
      }
      if (h.dKm < ev.minDist) { ev.minDist = h.dKm; ev.tcaMs = tAMs + h.tSec * 1000; ev.vrel = h.vrel; }
      ev.live = true; ev.lastSeen = nowRealMs;
    }
    for (const [key, ev] of this.events) {
      if (nowRealMs - ev.lastSeen > 250) ev.live = false;
      if (nowRealMs - ev.lastSeen > this.lingerMs) this.events.delete(key);
    }
  }
  active() { return Array.from(this.events.values()); }
}

// ============================================================================
// §6 · Risk assessment
// ↔ src/risk_assessment/collision_probability.py (CollisionProbabilityEngine
//   — Foster-style 2-D gaussian, simplified to a scalar here)
// ============================================================================
const HBR_KM = 0.02; // combined hard-body radius, 20 m

function positionSigmaKm(leadMs, sensorSigmaKm = 0) {
  // Tracking uncertainty grows the further ahead the twin looks. Public GP
  // elements are only good to roughly a kilometre, which is exactly why
  // ~1 km misses sit in the "manoeuvre warranted" probability band.
  return 0.5 + 0.15 * Math.max(leadMs, 0) / 3600000 + sensorSigmaKm;
}
function collisionProbability(missKm, sigmaKm) {
  const s2 = 2 * sigmaKm * sigmaKm;
  return clamp((HBR_KM * HBR_KM / s2) * Math.exp(-(missKm * missKm) / s2), 0, 1);
}
function fmtProb(p) {
  if (p <= 0) return "<1e-12";
  if (p < 1e-3) return p.toExponential(1).replace("e-", "e−");
  return p.toFixed(3);
}
function severity(p) {
  if (p >= 1e-4) return { cls: "crit",  label: "serious" };
  if (p >= 1e-5) return { cls: "high",  label: "elevated" };
  if (p >= 1e-6) return { cls: "watch", label: "watch" };
  return { cls: "low", label: "routine" };
}

// ============================================================================
// §7 · Sensor simulation — time-correlated tracking noise
// ↔ src/sensor_simulation/noise_models.py (CorrelatedNoiseModel)
//
// Ornstein–Uhlenbeck per axis: errors wander and decay (τ ≈ 90 s) instead of
// being redrawn every frame, so the sensor picture is plausibly wrong rather
// than vibrating (another v1 bug).
// ============================================================================
class SensorNoise {
  constructor(n, seed = 7) {
    this.off = new Float64Array(n * 3);
    this.rng = mulberry32(seed);
    this.tau = 90;
  }
  step(dtSec, sigmaKm) {
    const a = Math.exp(-dtSec / this.tau);
    const s = sigmaKm * Math.sqrt(1 - a * a);
    const o = this.off;
    for (let i = 0; i < o.length; i++) o[i] = o[i] * a + gaussOf(this.rng) * s;
  }
  apply(src, dst) {
    const o = this.off;
    for (let i = 0; i < o.length; i++) dst[i] = src[i] + o[i];
  }
}

// ============================================================================
// §8 · Manoeuvre planning — what-if burns, twin-side only
//
// An along-track impulse Δv produces the classic Clohessy–Wiltshire response
// relative to the unburned orbit:
//   radial      x(t) = (2Δv/n)(1 − cos nt)
//   along-track y(t) = (4Δv/n)·sin nt − 3Δv·t
// The −3Δv·t drift term is the working physics of collision avoidance: a burn
// of centimetres per second, given hours of lead, moves a satellite kilometres
// off its old timeline. (Counter-intuitive bonus: thrust forward → arrive later.)
// ============================================================================
function cwOffset(dvKmS, nRad, dtSec, out) {
  const nt = nRad * dtSec;
  out[0] = (2 * dvKmS / nRad) * (1 - Math.cos(nt));        // radial
  out[1] = (4 * dvKmS / nRad) * Math.sin(nt) - 3 * dvKmS * dtSec; // along-track
}

// Mutates pos by the CW offset expressed in the satellite's RSW frame.
function applyBurnOffset(tMs, pos, vel, burn) {
  if (tMs <= burn.tMs || burn.dvKmS === 0) return;
  const r = Math.hypot(pos[0], pos[1], pos[2]);
  const v2 = vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2];
  const aSma = 1 / (2 / r - v2 / MU);
  if (!(aSma > 0)) return;
  const nRad = Math.sqrt(MU / (aSma ** 3));
  cwOffset(burn.dvKmS, nRad, (tMs - burn.tMs) / 1000, _cw);
  // RSW basis from current state
  const rx = pos[0] / r, ry = pos[1] / r, rz = pos[2] / r;
  let wx = ry * vel[2] - rz * vel[1], wy = rz * vel[0] - rx * vel[2], wz = rx * vel[1] - ry * vel[0];
  const wm = Math.hypot(wx, wy, wz) || 1; wx /= wm; wy /= wm; wz /= wm;
  const sx = wy * rz - wz * ry, sy = wz * rx - wx * rz, sz = wx * ry - wy * rx;
  pos[0] += _cw[0] * rx + _cw[1] * sx;
  pos[1] += _cw[0] * ry + _cw[1] * sy;
  pos[2] += _cw[0] * rz + _cw[1] * sz;
}
const _cw = [0, 0];

// Exact two-body miss profile around a conjunction, optionally with a burn on
// sat A — powers the range-vs-time chart and the before/after readout.
function missProfile(satA, satB, t0Ms, t1Ms, stepSec, burn) {
  const pa = [0, 0, 0], va = [0, 0, 0], pb = [0, 0, 0], vb = [0, 0, 0];
  const samples = [];
  let minD = Infinity, minT = t0Ms;
  for (let t = t0Ms; t <= t1Ms; t += stepSec * 1000) {
    propagateECI(satA, t, pa, va);
    if (burn) applyBurnOffset(t, pa, va, burn);
    propagateECI(satB, t, pb, vb);
    const d = Math.hypot(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]);
    samples.push({ tMs: t, dKm: d });
    if (d < minD) { minD = d; minT = t; }
  }
  return { samples, minD, minT };
}

// ============================================================================
// §9 · Prediction scan — the twin running ahead of the clock
//
// Coarse time steps with full-fleet SGP4 + hash + linear refinement per window;
// hash radius (threshold + VREL_MAX·dt/2 each side) keeps the sieve complete
// for LEO closing speeds. Work is sliced into ~12 ms chunks so the page stays
// alive, and the twin globe visibly fast-forwards while it runs.
// ============================================================================
async function scanAhead({ fleet, t0Ms, horizonMs, stepSec = 45, screenKm = 30, onProgress, isCancelled }) {
  const n = fleet.length;
  const pos = new Float64Array(n * 3), vel = new Float64Array(n * 3);
  const win = new PropagationWindow(fleet);
  const radius = screenKm + (VREL_MAX * stepSec) / 2;
  const best = new Map();
  const steps = Math.ceil(horizonMs / (stepSec * 1000));
  let sliceStart = performance.now();

  for (let s = 0; s < steps; s++) {
    const t = t0Ms + s * stepSec * 1000;
    win.fill(t, pos, vel);
    const cand = candidatePairs(pos, n, radius);
    for (let k = 0; k < cand.length; k += 2) {
      const i = cand[k], j = cand[k + 1];
      const ca = closestApproach(pos, vel, i, j, stepSec);
      if (ca.dKm < screenKm && ca.vrel > 0.15) {     // vrel filter drops co-orbital formation neighbours
        const key = `${i}-${j}`;
        const prev = best.get(key);
        if (!prev || ca.dKm < prev.dKm) best.set(key, { i, j, dKm: ca.dKm, tcaMs: t + ca.tSec * 1000, vrel: ca.vrel });
      }
    }
    if (onProgress && s % 4 === 0) onProgress(s / steps, t, best.size);
    if (performance.now() - sliceStart > 20) {
      // setTimeout(0), not rAF: rAF can be throttled to ~1 Hz in unfocused
      // tabs, which once stretched this scan from <1 s to 20 s.
      await new Promise(r => setTimeout(r, 0));
      if (isCancelled && isCancelled()) return null;
      sliceStart = performance.now();
    }
  }

  // Exact refinement of the shortlist: fine sampling around each coarse TCA.
  const top = Array.from(best.values()).sort((a, b) => a.dKm - b.dKm).slice(0, 12);
  const out = [];
  for (const ev of top) {
    const prof = missProfile(fleet[ev.i], fleet[ev.j], ev.tcaMs - 90000, ev.tcaMs + 90000, 2);
    out.push({ a: ev.i, b: ev.j, dKm: prof.minD, tcaMs: prof.minT, vrel: ev.vrel });
  }
  out.sort((a, b) => a.dKm - b.dKm);
  if (onProgress) onProgress(1, t0Ms + horizonMs, out.length);
  return out;
}

// ============================================================================
// §10 · Scene — Earth, fleet, conjunction graphics
// ↔ dashboard/app.py (Plotly globe → Three.js)
// ============================================================================
function makeEarthTexture(landPolys) {
  const W = 2048, H = 1024;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  // Ocean
  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, "#0b1422"); grad.addColorStop(0.5, "#101c30"); grad.addColorStop(1, "#0b1422");
  ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);
  // Graticule
  ctx.strokeStyle = "rgba(143,213,192,0.07)";
  ctx.lineWidth = 1;
  for (let lon = -150; lon <= 180; lon += 30) {
    const x = ((lon + 180) / 360) * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = ((90 - lat) / 180) * H;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  // Land (Natural Earth 110m, equirectangular)
  if (landPolys) {
    ctx.fillStyle = "#243447";
    ctx.strokeStyle = "rgba(143,213,192,0.45)";
    ctx.lineWidth = 1.1;
    for (const poly of landPolys) {
      ctx.beginPath();
      for (let k = 0; k < poly.length; k += 2) {
        const x = ((poly[k] + 180) / 360) * W;
        const y = ((90 - poly[k + 1]) / 180) * H;
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath(); ctx.fill(); ctx.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function makeStarfield(count) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3), col = new Float32Array(count * 3);
  const rng = mulberry32(42);
  for (let i = 0; i < count; i++) {
    const u = rng() * 2 - 1, t = rng() * TWO_PI, r = 140, s = Math.sqrt(1 - u * u);
    pos[i * 3] = r * s * Math.cos(t); pos[i * 3 + 1] = r * u; pos[i * 3 + 2] = r * s * Math.sin(t);
    const b = 0.4 + rng() * 0.6;
    col[i * 3] = b; col[i * 3 + 1] = b; col[i * 3 + 2] = b * (0.92 + rng() * 0.16);
  }
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.16, sizeAttenuation: true, vertexColors: true, transparent: true, opacity: 0.8, depthWrite: false,
  }));
}

const MAX_SATS = 3000;

function buildScene(canvas, landPolys) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
  renderer.autoClear = false;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 1200);
  camera.position.set(0, 6, 21);

  const earthR = R_EARTH * SCENE_SCALE;

  // Earth group rotates by GMST so longitude lines up with the real planet.
  const earthGroup = new THREE.Group();
  scene.add(earthGroup);
  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(earthR, 96, 64),
    new THREE.MeshPhongMaterial({
      map: makeEarthTexture(landPolys),
      shininess: 14, specular: 0x16202e,
      emissive: 0x0a1322, emissiveIntensity: 0.55,
    })
  );
  earthGroup.add(earth);

  // Atmosphere rim
  const atmo = new THREE.Mesh(
    new THREE.SphereGeometry(earthR * 1.045, 48, 32),
    new THREE.ShaderMaterial({
      transparent: true, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false,
      uniforms: { uColor: { value: new THREE.Color(COL.mint) } },
      vertexShader: `varying vec3 vN,vV;void main(){vN=normalize(normalMatrix*normal);vec4 mv=modelViewMatrix*vec4(position,1.0);vV=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}`,
      fragmentShader: `varying vec3 vN,vV;uniform vec3 uColor;void main(){float r=pow(1.0-max(dot(vN,vV),0.0),4.5);gl_FragColor=vec4(uColor,r*0.32);}`,
    })
  );
  scene.add(atmo);

  scene.add(new THREE.AmbientLight(0x8a97ad, 0.55));
  const sun = new THREE.DirectionalLight(0xfff3dd, 1.25);
  sun.position.set(30, 12, 16);
  scene.add(sun);
  scene.add(makeStarfield(1200));

  // Fleet (instanced spheres)
  const sats = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.028, 6, 6),
    new THREE.MeshBasicMaterial({ color: 0xffffff }),
    MAX_SATS
  );
  sats.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  sats.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_SATS * 3), 3);
  sats.count = 0;
  scene.add(sats);

  // Conjunction flash lines + pulsing markers
  const flashGeo = new THREE.BufferGeometry();
  flashGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(300 * 6), 3));
  flashGeo.setDrawRange(0, 0);
  const flashes = new THREE.LineSegments(flashGeo, new THREE.LineBasicMaterial({ color: COL.coral, transparent: true, opacity: 0.9 }));
  scene.add(flashes);

  const pulses = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.11, 10, 10),
    new THREE.MeshBasicMaterial({ color: COL.coral, transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false }),
    600
  );
  pulses.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  pulses.count = 0;
  scene.add(pulses);

  // Selected-pair orbit lines + closest-approach marker
  const orbitA = makeOrbitLine(COL.mintHi);
  const orbitB = makeOrbitLine(COL.blue);
  const orbitGhost = makeOrbitLine(COL.coral, 0.35, true);
  scene.add(orbitA.line, orbitB.line, orbitGhost.line);

  // Closest-approach reticle: a small core + a thin ring that always faces the
  // camera (rotated each frame by the caller via lookAt-style billboarding).
  const tcaMarker = new THREE.Group();
  tcaMarker.add(new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 10, 10),
    new THREE.MeshBasicMaterial({ color: COL.coral, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false })
  ));
  tcaMarker.add(new THREE.Mesh(
    new THREE.RingGeometry(0.16, 0.185, 40),
    new THREE.MeshBasicMaterial({ color: COL.coral, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false })
  ));
  tcaMarker.visible = false;
  scene.add(tcaMarker);

  return { renderer, scene, camera, earthGroup, sats, flashes, flashGeo, pulses, orbitA, orbitB, orbitGhost, tcaMarker, earthR };
}

function makeOrbitLine(color, opacity = 0.55, dashed = false) {
  const N = 160;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array((N + 1) * 3), 3));
  geo.setDrawRange(0, 0);
  const mat = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.18, gapSize: 0.12 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(geo, mat);
  line.visible = false;
  return { line, geo, N, dashed };
}

// Sample one full orbit of a satellite (optionally with a burn) into a line.
function traceOrbit(orbit, sat, tcaMs, burn) {
  const periodMs = orbitalPeriodMs(sat);
  const attr = orbit.geo.attributes.position;
  const p = [0, 0, 0], v = [0, 0, 0];
  for (let k = 0; k <= orbit.N; k++) {
    const t = tcaMs - periodMs / 2 + (periodMs * k) / orbit.N;
    propagateECI(sat, t, p, v);
    if (burn) applyBurnOffset(t, p, v, burn);
    attr.array[k * 3]     = p[0] * SCENE_SCALE;
    attr.array[k * 3 + 1] = p[2] * SCENE_SCALE;
    attr.array[k * 3 + 2] = -p[1] * SCENE_SCALE;
  }
  attr.needsUpdate = true;
  orbit.geo.setDrawRange(0, orbit.N + 1);
  orbit.geo.computeBoundingSphere();
  if (orbit.dashed) orbit.line.computeLineDistances();
  orbit.line.visible = true;
}

// ============================================================================
// §11 · Camera — drag orbit, wheel/pinch zoom, eased focus
// ============================================================================
function makeCamera(camera, dom, reducedMotion) {
  const st = {
    target: new THREE.Vector3(),
    distance: camera.position.length(),
    az: Math.atan2(camera.position.x, camera.position.z),
    el: Math.asin(camera.position.y / camera.position.length()),
    auto: !reducedMotion,
    pointers: new Map(),
    pinchDist: 0,
    tween: null,
    chase: null,   // fn() → Vector3, followed every frame
  };
  const apply = () => {
    const ce = Math.cos(st.el);
    camera.position.set(
      st.target.x + st.distance * ce * Math.sin(st.az),
      st.target.y + st.distance * Math.sin(st.el),
      st.target.z + st.distance * ce * Math.cos(st.az)
    );
    camera.lookAt(st.target);
  };
  apply();

  dom.addEventListener("pointerdown", e => {
    dom.setPointerCapture(e.pointerId);
    st.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (st.pointers.size === 2) {
      const [a, b] = [...st.pointers.values()];
      st.pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
    }
    st.auto = false; st.tween = null;
  });
  dom.addEventListener("pointermove", e => {
    const p = st.pointers.get(e.pointerId);
    if (!p) return;
    if (st.pointers.size === 1) {
      st.az -= (e.clientX - p.x) * 0.006;
      st.el = clamp(st.el + (e.clientY - p.y) * 0.006, -1.25, 1.25);
      apply();
    }
    p.x = e.clientX; p.y = e.clientY;
    if (st.pointers.size === 2) {
      const [a, b] = [...st.pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (st.pinchDist > 0) {
        st.distance = clamp(st.distance * (st.pinchDist / d), 7.6, 70);
        apply();
      }
      st.pinchDist = d;
    }
  });
  const release = e => { st.pointers.delete(e.pointerId); st.pinchDist = 0; };
  dom.addEventListener("pointerup", release);
  dom.addEventListener("pointercancel", release);
  dom.addEventListener("wheel", e => {
    e.preventDefault();
    st.auto = false;
    st.distance = clamp(st.distance * Math.exp(e.deltaY * 0.0011), 7.6, 70);
    apply();
  }, { passive: false });

  return {
    state: st,
    tick(dt, nowMs) {
      if (st.chase) {
        const tgt = st.chase();
        if (tgt) st.target.lerp(tgt, Math.min(1, dt * 4));
      }
      if (st.tween) {
        const k = clamp((nowMs - st.tween.t0) / st.tween.dur, 0, 1);
        const e2 = easeOut(k);
        st.target.lerpVectors(st.tween.fromT, st.tween.toT, e2);
        st.distance = lerp(st.tween.fromD, st.tween.toD, e2);
        st.az = lerp(st.tween.fromAz, st.tween.toAz, e2);
        st.el = lerp(st.tween.fromEl, st.tween.toEl, e2);
        if (k >= 1) st.tween = null;
      } else if (st.auto && !reducedMotion) {
        st.az += dt * 0.04;
      }
      apply();
    },
    focus(targetVec, distance, nowMs) {
      const toAz = Math.atan2(targetVec.x, targetVec.z);
      st.tween = {
        t0: nowMs, dur: reducedMotion ? 1 : 1100,
        fromT: st.target.clone(), toT: targetVec.clone(),
        fromD: st.distance, toD: distance,
        fromAz: st.az, toAz: st.az + Math.atan2(Math.sin(toAz - st.az), Math.cos(toAz - st.az)),
        fromEl: st.el, toEl: clamp(Math.asin(targetVec.y / (targetVec.length() || 1)), -1.1, 1.1),
      };
      st.auto = false;
    },
    resetView(nowMs) {
      st.chase = null;
      this.focus(new THREE.Vector3(0, 0, 0), 21, nowMs);
      st.auto = !reducedMotion;
    },
    setChase(fn) { st.chase = fn; },
  };
}

// ============================================================================
// §12 · UI template — the whole demo DOM is built here so both host pages
// (website + repo) stay trivially identical.
// ============================================================================
const STAGES = ["live", "twin", "predict", "act", "explore"];
const STAGE_META = {
  live:    { n: 1, label: "Live sky" },
  twin:    { n: 2, label: "Spin up the twin" },
  predict: { n: 3, label: "Predict" },
  act:     { n: 4, label: "Act" },
  explore: { n: 5, label: "Explore" },
};

function buildUI(root) {
  root.classList.add("twin");
  root.innerHTML = `
  <div class="twin__stepper" role="tablist" aria-label="Digital twin tour">
    ${STAGES.map(s => `
      <button class="twin__step" data-tw-step="${s}" role="tab" aria-selected="false">
        <span class="twin__step-n">${STAGE_META[s].n}</span>
        <span class="twin__step-label">${STAGE_META[s].label}</span>
      </button>`).join('<span class="twin__step-sep" aria-hidden="true"></span>')}
  </div>

  <div class="twin__stage">
    <div class="twin__viewport" data-tw-viewport>
      <canvas class="twin__canvas" data-tw-canvas aria-label="3D view of satellites around Earth"></canvas>
      <div class="twin__labels" data-tw-labels aria-hidden="true"></div>

      <div class="twin__split" data-tw-split hidden>
        <div class="twin__split-half">
          <span class="twin__split-tag twin__split-tag--real">REALITY · always 1×</span>
          <span class="twin__split-clock" data-tw-clock-real>—</span>
        </div>
        <div class="twin__split-divider"></div>
        <div class="twin__split-half">
          <span class="twin__split-tag twin__split-tag--twin" data-tw-twin-tag>DIGITAL TWIN</span>
          <span class="twin__split-clock" data-tw-clock-twin>—</span>
        </div>
      </div>

      <div class="twin__hud" data-tw-hud>
        <span class="twin__chip" data-tw-chip-clock title="Current UTC time">— UTC</span>
        <span class="twin__chip" data-tw-chip-data>loading…</span>
        <span class="twin__chip" data-tw-chip-count hidden></span>
        <span class="twin__chip twin__chip--accent" data-tw-chip-closest hidden></span>
        <span class="twin__chip" data-tw-chip-fps hidden></span>
      </div>

      <div class="twin__scanbar" data-tw-scanbar hidden>
        <div class="twin__scanbar-head">
          <span class="twin__scanbar-dot"></span>
          <span data-tw-scan-label>Twin scanning ahead…</span>
        </div>
        <div class="twin__scanbar-track"><div class="twin__scanbar-fill" data-tw-scan-fill></div></div>
      </div>

      <div class="twin__loading" data-tw-loading>
        <span class="twin__loading-dot"></span>
        <span data-tw-loading-text>Contacting CelesTrak…</span>
      </div>

      <p class="twin__hint" data-tw-hint>drag to orbit · scroll or pinch to zoom</p>
    </div>

    <aside class="twin__panel" data-tw-panel>

      <div class="twin__card" data-tw-card="live">
        <p class="twin__kicker">Step 1 · The real sky, right now</p>
        <p class="twin__copy">Every dot is a <strong>real satellite</strong>, drawn where it
          actually is at this second — positions come from the public radar catalog
          (CelesTrak) and the clock is your clock. The tagged one is the
          <strong>International Space Station</strong>.</p>
        <p class="twin__copy twin__copy--dim">Reality has one annoying property: it only plays
          forward, at 1×. To see trouble coming, operators build a copy of the sky in software —
          a <em>digital twin</em>.</p>
        <button class="twin__btn twin__btn--primary" data-tw-cta="live">Spin up the digital twin →</button>
      </div>

      <div class="twin__card" data-tw-card="twin">
        <p class="twin__kicker">Step 2 · Reality and its twin</p>
        <p class="twin__copy"><strong>Left:</strong> the real sky, still ticking at 1×.
          <strong>Right:</strong> its digital twin — same satellites, same physics, but free of
          the calendar. Drag the slider to push the twin into the future. Reality doesn't follow.</p>
        <div class="twin__slider">
          <div class="twin__slider-head">
            <span class="label">Twin clock offset</span>
            <span class="value" data-tw-scrub-label>+0s</span>
          </div>
          <input type="range" min="-60" max="360" step="5" value="0" data-tw-scrub aria-label="Twin clock offset in minutes">
        </div>
        <p class="twin__copy twin__copy--dim">This gap between the two clocks is the whole trick:
          anything the twin meets at <em>+3 h</em> is something reality hasn't lived yet.</p>
        <button class="twin__btn twin__btn--primary" data-tw-cta="twin">Scan the next 3 hours →</button>
      </div>

      <div class="twin__card" data-tw-card="predict">
        <p class="twin__kicker">Step 3 · The twin flies the future first</p>
        <p class="twin__copy" data-tw-predict-summary>The twin is sweeping the next three hours of
          orbits, looking for pairs that pass dangerously close…</p>
        <ol class="twin__events" data-tw-events></ol>
        <p class="twin__copy twin__copy--dim" data-tw-predict-foot hidden>Tap a pass to aim the
          twin's camera at it; <strong>▶ watch</strong> replays it at high speed.</p>
        <button class="twin__btn twin__btn--primary" data-tw-cta="predict" disabled>Test an avoidance manoeuvre →</button>
      </div>

      <div class="twin__card" data-tw-card="act">
        <p class="twin__kicker">Step 4 · Rehearse the fix on the twin</p>
        <div class="twin__act" data-tw-act-detail>
          <p class="twin__copy">No close pass selected yet.</p>
        </div>
        <button class="twin__btn twin__btn--primary" data-tw-cta="act">Open the sandbox →</button>
      </div>

      <div class="twin__card" data-tw-card="explore">
        <p class="twin__kicker">Step 5 · Your console now</p>

        <div class="twin__group">
          <span class="twin__group-label">Twin playback</span>
          <div class="twin__seg" role="group" aria-label="Twin speed">
            <button data-tw-speed="1" aria-pressed="true">1×</button>
            <button data-tw-speed="60" aria-pressed="false">60×</button>
            <button data-tw-speed="600" aria-pressed="false">600×</button>
          </div>
          <div class="twin__row">
            <button class="twin__btn" data-tw-resync title="Snap the twin back to the real clock">⟲ Re-sync with reality</button>
            <button class="twin__btn" data-tw-rescan>Scan ahead again</button>
          </div>
        </div>

        <div class="twin__group">
          <span class="twin__group-label">Alert threshold <span class="twin__group-value" data-tw-threshold-label>5.0 km</span></span>
          <input type="range" min="1" max="20" step="0.5" value="5" data-tw-threshold aria-label="Conjunction alert threshold (km)">
        </div>

        <div class="twin__group">
          <span class="twin__group-label">Constellation</span>
          <select class="twin__select" data-tw-source aria-label="Data source">
            <option value="active" selected>Active LEO catalog</option>
            <option value="starlink">Starlink</option>
            <option value="oneweb">OneWeb</option>
            <option value="synthetic">Synthetic fleet (offline)</option>
          </select>
          <select class="twin__select" data-tw-count aria-label="Fleet size">
            <option value="500">500 objects</option>
            <option value="1000">1,000 objects</option>
            <option value="1500">1,500 objects</option>
            <option value="2500">2,500 objects</option>
          </select>
          <p class="twin__srcinfo" data-tw-srcinfo></p>
        </div>

        <div class="twin__group">
          <label class="twin__toggle">
            <input type="checkbox" data-tw-noise-toggle>
            <span>Researcher mode — imperfect sensors</span>
          </label>
          <div class="twin__noisectl" data-tw-noisectl hidden>
            <span class="twin__group-label">Tracking noise (1σ) <span class="twin__group-value" data-tw-noise-label>300 m</span></span>
            <input type="range" min="50" max="2000" step="50" value="300" data-tw-noise aria-label="Sensor noise sigma (metres)">
            <p class="twin__copy twin__copy--dim">The twin now sees the sky through wobbly
              instruments. <strong class="t-coral">MISSED</strong> = a real close pass the noisy twin
              didn't flag · <strong class="t-coral-dim">FALSE+</strong> = a phantom alert.
              <span data-tw-noise-stats></span></p>
          </div>
        </div>

        <div class="twin__group">
          <span class="twin__group-label">Close passes <span class="twin__group-value" data-tw-live-count></span></span>
          <ol class="twin__events twin__events--compact" data-tw-events-live></ol>
        </div>

        <button class="twin__btn" data-tw-restart>↺ Restart the tour</button>
      </div>

    </aside>
  </div>

  <p class="twin__foot">
    <span data-tw-foot-data>Orbit data: CelesTrak general-perturbations catalog, propagated with SGP4 in your browser.</span>
    <span class="twin__foot-sep">·</span>
    <span>Screening is simplified vs. operational practice (point estimates, no covariance) — same architecture, honest shortcuts.</span>
  </p>`;

  const $ = sel => root.querySelector(sel);
  return {
    root,
    viewport: $("[data-tw-viewport]"), canvas: $("[data-tw-canvas]"), labels: $("[data-tw-labels]"),
    split: $("[data-tw-split]"), clockReal: $("[data-tw-clock-real]"), clockTwin: $("[data-tw-clock-twin]"),
    twinTag: $("[data-tw-twin-tag]"),
    hud: $("[data-tw-hud]"),
    chipClock: $("[data-tw-chip-clock]"), chipData: $("[data-tw-chip-data]"),
    chipCount: $("[data-tw-chip-count]"), chipClosest: $("[data-tw-chip-closest]"), chipFps: $("[data-tw-chip-fps]"),
    scanbar: $("[data-tw-scanbar]"), scanLabel: $("[data-tw-scan-label]"), scanFill: $("[data-tw-scan-fill]"),
    loading: $("[data-tw-loading]"), loadingText: $("[data-tw-loading-text]"),
    hint: $("[data-tw-hint]"),
    panel: $("[data-tw-panel]"),
    steps: Array.from(root.querySelectorAll("[data-tw-step]")),
    cards: Object.fromEntries(STAGES.map(s => [s, $(`[data-tw-card="${s}"]`)])),
    ctas: Object.fromEntries(STAGES.map(s => [s, $(`[data-tw-cta="${s}"]`)])),
    scrub: $("[data-tw-scrub]"), scrubLabel: $("[data-tw-scrub-label]"),
    events: $("[data-tw-events]"), eventsLive: $("[data-tw-events-live]"),
    predictSummary: $("[data-tw-predict-summary]"), predictFoot: $("[data-tw-predict-foot]"),
    actDetail: $("[data-tw-act-detail]"),
    speedBtns: Array.from(root.querySelectorAll("[data-tw-speed]")),
    resync: $("[data-tw-resync]"), rescan: $("[data-tw-rescan]"), restart: $("[data-tw-restart]"),
    threshold: $("[data-tw-threshold]"), thresholdLabel: $("[data-tw-threshold-label]"),
    source: $("[data-tw-source]"), count: $("[data-tw-count]"), srcinfo: $("[data-tw-srcinfo]"),
    noiseToggle: $("[data-tw-noise-toggle]"), noisectl: $("[data-tw-noisectl]"),
    noise: $("[data-tw-noise]"), noiseLabel: $("[data-tw-noise-label]"), noiseStats: $("[data-tw-noise-stats]"),
    liveCount: $("[data-tw-live-count]"),
    footData: $("[data-tw-foot-data]"),
  };
}

// ============================================================================
// §13 · The simulator — state, stages, main loop
// ============================================================================
export function startDigitalTwin(root, opts = {}) {
  const dataBase = opts.dataBase ?? root.dataset.twinDataBase ?? "../assets/data/";
  const apiBase  = opts.apiBase  ?? (root.dataset.twinApiBase === "" ? null : (root.dataset.twinApiBase ?? "/api"));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const small = Math.min(window.innerWidth, window.innerHeight) < 720 || (navigator.hardwareConcurrency || 8) <= 4;

  const ui = buildUI(root);

  let sc;
  try {
    sc = buildScene(ui.canvas, null); // texture redrawn once coastlines arrive
  } catch (err) {
    console.error("[digital-twin] WebGL unavailable:", err);
    ui.loadingText.textContent = "This demo needs WebGL — your browser or device has it disabled.";
    return () => {};
  }
  const cam = makeCamera(sc.camera, ui.canvas, reducedMotion);

  // Coastlines load async, then the texture is swapped in place.
  fetch(`${dataBase}land-110m.json`).then(r => r.json()).then(land => {
    const tex = makeEarthTexture(land.polys);
    sc.earthGroup.children[0].material.map = tex;
    sc.earthGroup.children[0].material.needsUpdate = true;
  }).catch(() => {});

  // ----- Simulation state ----------------------------------------------------
  const S = {
    stage: "live",
    fleet: [],
    n: 0,
    catalogTotal: 0,
    live: false,
    fetchedAt: null,
    sourceKey: "active",
    realWin: null,         // PropagationWindow — the mirror, always at wall clock
    twinWin: null,         // PropagationWindow — the twin, offset/speed adjustable
    twinTimeMs: Date.now(),
    twinSpeed: 1,
    scrubOffsetMin: 0,
    thresholdKm: 5,
    tracker: new ConjunctionTracker(),       // twin, clean
    trackerNoisy: new ConjunctionTracker(),  // twin, sensor view (researcher mode)
    noiseOn: false,
    noiseSigmaKm: 0.3,
    noise: null,
    noisyPos: null,
    closestNow: null,      // {a, b, dKm} from the reality window
    predicted: null,       // events from scanAhead
    selected: null,        // selected predicted event
    burn: null,            // {satIdx, tMs, dvKmS}
    watching: null,        // {untilMs} — twin replay of a pass
    scanning: false,
    scanId: 0,
    renderPos: null,       // scratch Float64Array for sampling
    renderPosNoisy: null,
    disposed: false,
  };

  const setLoading = msg => {
    ui.loading.hidden = !msg;
    if (msg) ui.loadingText.textContent = msg;
  };

  // ----- Fleet loading ---------------------------------------------------------
  async function loadFleet() {
    const want = +ui.count.value || (small ? 600 : 1000);
    S.predicted = null; S.selected = null; S.burn = null; S.watching = null;
    S.tracker.reset(); S.trackerNoisy.reset();
    renderEventList(); renderActDetail();

    if (S.sourceKey === "synthetic") {
      applyFleet(generateSyntheticFleet(want, (Math.random() * 1e6) | 0), { live: false, fetchedAt: Date.now(), total: want, synthetic: true });
      return;
    }
    setLoading("Fetching the live satellite catalog…");
    const cat = await loadCatalog(S.sourceKey, { apiBase, dataBase });
    if (S.disposed) return;
    if (!cat) {
      ui.chipData.textContent = "offline — synthetic fleet";
      applyFleet(generateSyntheticFleet(want, 1), { live: false, fetchedAt: null, total: want, synthetic: true });
      return;
    }
    applyFleet(buildFleet(cat, want), { live: cat.live, fetchedAt: cat.fetchedAt, total: cat.main.length + cat.stations.length });
  }

  function applyFleet(fleet, meta) {
    S.fleet = fleet;
    S.n = fleet.length;
    S.live = !!meta.live;
    S.fetchedAt = meta.fetchedAt;
    S.catalogTotal = meta.total;
    S.synthetic = !!meta.synthetic;
    S.realWin = new PropagationWindow(fleet);
    S.twinWin = new PropagationWindow(fleet);
    S.twinTimeMs = Date.now() + S.scrubOffsetMin * 60000;
    S.renderPos = new Float64Array(S.n * 3);
    S.renderPosNoisy = new Float64Array(S.n * 3);
    S.noise = new SensorNoise(S.n);
    S.noisyPos = new Float64Array(S.n * 3);
    sc.sats.count = S.n;
    S.twinWin.onSubstep = onTwinSubstep;
    S.realWin.onSubstep = onRealSubstep;
    setLoading(null);
    updateDataChip();
    buildLabels();
    setLoading(null);
  }

  function updateDataChip() {
    let txt, cls = "";
    if (S.synthetic) {
      txt = "SYNTHETIC fleet (offline mode)";
    } else if (S.live) {
      txt = `LIVE · TLEs ${S.fetchedAt ? fmtAge(Date.now() - S.fetchedAt) : "fresh"}`;
      cls = "twin__chip--live";
    } else {
      txt = `SNAPSHOT · ${S.fetchedAt ? fmtAge(Date.now() - S.fetchedAt) : "bundled"}`;
    }
    ui.chipData.textContent = txt;
    ui.chipData.className = `twin__chip ${cls}`;
    ui.chipCount.hidden = false;
    ui.chipCount.textContent = `${S.n.toLocaleString()} of ${S.catalogTotal.toLocaleString()} objects`;
    ui.srcinfo.innerHTML = S.synthetic
      ? "Synthetic Kepler+J2 fleet — no network needed."
      : `${SOURCES[S.sourceKey]?.label || "Catalog"} · showing <strong>${S.n.toLocaleString()}</strong> of ${S.catalogTotal.toLocaleString()} catalogued ·
         ${S.live ? "live via CelesTrak" : "bundled snapshot"} · <a href="https://celestrak.org" target="_blank" rel="noopener">celestrak.org</a>`;
    ui.footData.textContent = S.synthetic
      ? "Offline mode: synthetic fleet with the same physics pipeline."
      : `Orbit data: CelesTrak GP catalog (${S.live ? "fetched live" : "bundled snapshot"}${S.fetchedAt ? ", " + fmtAge(Date.now() - S.fetchedAt) : ""}), propagated with SGP4 in your browser.`;
  }

  // ----- Screening hooks -------------------------------------------------------
  function onTwinSubstep(tA, tB, posA, velA, dtSec) {
    const hits = screenSubstep(posA, velA, S.n, S.thresholdKm, dtSec);
    S.tracker.update(hits, tA, performance.now());
    if (S.noiseOn) {
      S.noise.step(dtSec, S.noiseSigmaKm);
      S.noise.apply(posA, S.noisyPos);
      const noisyHits = screenSubstep(S.noisyPos, velA, S.n, S.thresholdKm, dtSec);
      S.trackerNoisy.update(noisyHits, tA, performance.now());
    }
  }
  function onRealSubstep(tA, tB, posA, velA, dtSec) {
    // Nearest-pair ticker for the live stage — generous fixed radius. Skip
    // co-moving pairs (docked vehicles, formations): they aren't "passes".
    const cand = candidatePairs(posA, S.n, 120);
    let best = null;
    for (let k = 0; k < cand.length; k += 2) {
      const i = cand[k], j = cand[k + 1];
      const dvx = velA[j * 3] - velA[i * 3], dvy = velA[j * 3 + 1] - velA[i * 3 + 1], dvz = velA[j * 3 + 2] - velA[i * 3 + 2];
      if (dvx * dvx + dvy * dvy + dvz * dvz < VREL_MIN * VREL_MIN) continue;
      const dx = posA[j * 3] - posA[i * 3], dy = posA[j * 3 + 1] - posA[i * 3 + 1], dz = posA[j * 3 + 2] - posA[i * 3 + 2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (!best || d < best.dKm) best = { a: i, b: j, dKm: d };
    }
    S.closestNow = best;
  }

  // ----- Labels ----------------------------------------------------------------
  let labelEls = [];
  function buildLabels() {
    ui.labels.innerHTML = "";
    labelEls = [];
    for (const sat of S.fleet) {
      if (!sat.famous) continue;
      const el = document.createElement("span");
      el.className = "twin__label";
      el.textContent = sat.famous;
      ui.labels.appendChild(el);
      labelEls.push({ el, idx: sat.id });
    }
  }
  const _proj = new THREE.Vector3();
  function placeLabels(positions, vpX, vpW, cssW, cssH) {
    for (const { el, idx } of labelEls) {
      const x = positions[idx * 3] * SCENE_SCALE, y = positions[idx * 3 + 2] * SCENE_SCALE, z = -positions[idx * 3 + 1] * SCENE_SCALE;
      _proj.set(x, y, z);
      // Occlusion: hide when the globe is between the camera and the satellite.
      const camPos = sc.camera.position;
      const d = _proj.clone().sub(camPos);
      const tClosest = clamp(-camPos.dot(d) / d.lengthSq(), 0, 1);
      const closest = camPos.clone().addScaledVector(d, tClosest);
      const occluded = tClosest < 1 && closest.length() < sc.earthR * 0.99;
      _proj.project(sc.camera);
      const sx = vpX + (_proj.x * 0.5 + 0.5) * vpW;
      const sy = (-_proj.y * 0.5 + 0.5) * cssH;
      const visible = !occluded && _proj.z < 1 && sx >= vpX && sx <= vpX + vpW;
      el.style.display = visible ? "" : "none";
      if (visible) el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px)`;
    }
  }

  // ----- Event list rendering ----------------------------------------------------
  function eventRow(ev, i, { withWatch }) {
    const a = S.fleet[ev.a], b = S.fleet[ev.b];
    const lead = ev.tcaMs - Date.now();
    const pc = collisionProbability(ev.dKm, positionSigmaKm(lead, S.noiseOn ? S.noiseSigmaKm : 0));
    const sev = severity(pc);
    const sel = S.selected && S.selected.a === ev.a && S.selected.b === ev.b;
    return `<li class="twin__event${sel ? " twin__event--sel" : ""}" data-tw-ev="${i}">
      <button class="twin__event-main" data-tw-ev-pick="${i}">
        <span class="twin__event-rank">${i + 1}</span>
        <span class="twin__event-body">
          <span class="twin__event-pair">${escapeHTML(a.name)} <span class="arr">↔</span> ${escapeHTML(b.name)}</span>
          <span class="twin__event-meta">
            <span class="twin__sev twin__sev--${sev.cls}"></span>${fmtDist(ev.dKm)} apart
            · ${fmtUTC(ev.tcaMs, false)} UTC · <span data-tw-cd="${ev.tcaMs}">${fmtCountdown(lead)}</span>
          </span>
        </span>
      </button>
      ${withWatch ? `<button class="twin__event-watch" data-tw-ev-watch="${i}" title="Replay this pass in the twin at 25×">▶ watch</button>` : ""}
    </li>`;
  }

  function renderEventList() {
    const evs = S.predicted || [];
    if (S.scanning) return;
    if (!evs.length) {
      ui.events.innerHTML = `<li class="twin__event-empty">No scan yet.</li>`;
      ui.eventsLive.innerHTML = `<li class="twin__event-empty">Run a scan, or crank the twin speed and watch for flashes.</li>`;
      return;
    }
    const rows = evs.slice(0, 7).map((ev, i) => eventRow(ev, i, { withWatch: true })).join("");
    ui.events.innerHTML = rows;
    ui.eventsLive.innerHTML = rows;
    ui.predictFoot.hidden = false;
  }

  function renderNoiseBadges() {
    if (!S.noiseOn) { ui.noiseStats.textContent = ""; return; }
    const clean = S.tracker.active().filter(e => e.live);
    const noisy = new Set(S.trackerNoisy.active().filter(e => e.live).map(e => e.key));
    const cleanKeys = new Set(clean.map(e => e.key));
    let missed = 0, falsePos = 0;
    for (const k of cleanKeys) if (!noisy.has(k)) missed++;
    for (const k of noisy) if (!cleanKeys.has(k)) falsePos++;
    ui.noiseStats.innerHTML = `Right now: <strong class="t-coral">${missed} missed</strong>, <strong class="t-coral-dim">${falsePos} phantom</strong>.`;
  }

  // ----- Selection / focus / watch -----------------------------------------------
  function selectEvent(i) {
    const ev = (S.predicted || [])[i];
    if (!ev) return;
    S.selected = ev;
    S.burn = null;
    S.twinWin.postHooks.clear();
    drawSelectedOrbits();
    focusOnEvent(ev);
    renderEventList();
    renderActDetail();
    ui.ctas.predict.disabled = false;
  }

  function focusOnEvent(ev) {
    const p = [0, 0, 0];
    propagateECI(S.fleet[ev.a], ev.tcaMs, p);
    const tgt = new THREE.Vector3(p[0] * SCENE_SCALE, p[2] * SCENE_SCALE, -p[1] * SCENE_SCALE);
    sc.tcaMarker.position.copy(tgt);
    sc.tcaMarker.visible = true;
    cam.focus(tgt, Math.max(4.6, ev.dKm * SCENE_SCALE * 40 + 3.6), performance.now());
  }

  function drawSelectedOrbits() {
    if (!S.selected) {
      sc.orbitA.line.visible = sc.orbitB.line.visible = sc.orbitGhost.line.visible = false;
      sc.tcaMarker.visible = false;
      return;
    }
    const { a, b, tcaMs } = S.selected;
    traceOrbit(sc.orbitB, S.fleet[b], tcaMs, null);
    if (S.burn && S.burn.dvKmS !== 0) {
      traceOrbit(sc.orbitGhost, S.fleet[a], tcaMs, null);          // old path, dashed coral
      traceOrbit(sc.orbitA, S.fleet[a], tcaMs, S.burn);            // new path, mint
    } else {
      sc.orbitGhost.line.visible = false;
      traceOrbit(sc.orbitA, S.fleet[a], tcaMs, null);
    }
  }

  function watchEvent(i) {
    const ev = (S.predicted || [])[i];
    if (!ev) return;
    S.selected = ev;
    drawSelectedOrbits();
    renderActDetail();
    S.twinTimeMs = ev.tcaMs - 50000;
    S.twinSpeed = 25;
    S.watching = { untilMs: ev.tcaMs + 25000 };
    // Chase the pair midpoint while the pass plays out.
    const pa = [0, 0, 0], va = [0, 0, 0], pb = [0, 0, 0];
    cam.setChase(() => {
      propagateECI(S.fleet[ev.a], S.twinTimeMs, pa, va);
      if (S.burn) applyBurnOffset(S.twinTimeMs, pa, va, S.burn);
      propagateECI(S.fleet[ev.b], S.twinTimeMs, pb);
      return new THREE.Vector3(
        (pa[0] + pb[0]) / 2 * SCENE_SCALE,
        (pa[2] + pb[2]) / 2 * SCENE_SCALE,
        -(pa[1] + pb[1]) / 2 * SCENE_SCALE
      );
    });
    cam.state.distance = Math.min(cam.state.distance, 4.2);
    ui.ctas.predict.disabled = false;
  }

  // ----- ACT panel -----------------------------------------------------------------
  function renderActDetail() {
    const ev = S.selected;
    if (!ev) {
      ui.actDetail.innerHTML = `<p class="twin__copy">Pick a close pass in step 3 first.</p>`;
      return;
    }
    const a = S.fleet[ev.a], b = S.fleet[ev.b];
    const dv = S.burn ? S.burn.dvKmS * 1e5 : 0;   // cm/s
    const burnT = Date.now() + 120000;            // burn two minutes from now
    const lead = ev.tcaMs - burnT;
    let after = ev.dKm;
    if (S.burn && S.burn.dvKmS !== 0) {
      const prof = missProfile(a, b, ev.tcaMs - 600000, ev.tcaMs + 600000, 4, S.burn);
      after = prof.minD;
    }
    const pcBefore = collisionProbability(ev.dKm, positionSigmaKm(ev.tcaMs - Date.now()));
    const pcAfter  = collisionProbability(after,  positionSigmaKm(ev.tcaMs - Date.now()));
    const safe = pcAfter < 1e-6 || after >= S.thresholdKm;
    const sevB = severity(pcBefore), sevA = severity(pcAfter);
    ui.actDetail.innerHTML = `
      <p class="twin__copy"><strong>${escapeHTML(a.name)}</strong> meets <strong>${escapeHTML(b.name)}</strong>
        at ${fmtUTC(ev.tcaMs, false)} UTC (<span data-tw-cd="${ev.tcaMs}">${fmtCountdown(ev.tcaMs - Date.now())}</span>),
        missing by <strong>${fmtDist(ev.dKm)}</strong> — at a closing speed of ${ev.vrel.toFixed(1)} km/s.</p>
      <div class="twin__slider">
        <div class="twin__slider-head">
          <span class="label">Nudge ${escapeHTML(shortName(a.name))} along-track</span>
          <span class="value">${dv === 0 ? "no burn" : (dv > 0 ? "+" : "−") + Math.abs(dv).toFixed(1) + " cm/s"}</span>
        </div>
        <input type="range" min="-15" max="15" step="0.5" value="${dv}" data-tw-dv aria-label="Burn size in centimetres per second">
      </div>
      <div class="twin__missrow">
        <div class="twin__miss">
          <span class="twin__miss-k">without burn</span>
          <span class="twin__miss-v">${fmtDist(ev.dKm)}</span>
          <span class="twin__miss-p"><span class="twin__sev twin__sev--${sevB.cls}"></span> ${sevB.label} · P≈${fmtProb(pcBefore)}</span>
        </div>
        <span class="twin__miss-arrow">→</span>
        <div class="twin__miss ${dv !== 0 ? (safe ? "twin__miss--safe" : "twin__miss--warn") : ""}">
          <span class="twin__miss-k">with burn</span>
          <span class="twin__miss-v">${dv === 0 ? "—" : fmtDist(after)}</span>
          <span class="twin__miss-p">${dv === 0 ? "drag the slider" : `<span class="twin__sev twin__sev--${sevA.cls}"></span> ${sevA.label} · P≈${fmtProb(pcAfter)}`}</span>
        </div>
      </div>
      <div class="twin__profile" data-tw-profile>${profileSVG(ev)}</div>
      <p class="twin__copy twin__copy--dim">A few <em>centimetres per second</em>, ${fmtCountdown(lead).replace("in ", "")} of
        lead time, and the pass opens up by kilometres — burning <em>earlier</em> is exponentially cheaper
        than burning late. The real ${escapeHTML(shortName(a.name))} hasn't moved: you rehearsed this on the twin.</p>`;
    const dvSlider = ui.actDetail.querySelector("[data-tw-dv]");
    dvSlider.addEventListener("input", () => {
      const cmS = +dvSlider.value;
      S.burn = cmS === 0 ? null : { satIdx: ev.a, tMs: burnT, dvKmS: cmS / 1e5 };
      S.twinWin.setPostHook(ev.a, S.burn ? (t, p, v) => applyBurnOffset(t, p, v, S.burn) : null);
      drawSelectedOrbits();
      renderActDetail();
    });
  }

  function shortName(n) { return n.length > 14 ? n.slice(0, 13) + "…" : n; }

  function profileSVG(ev) {
    const W = 280, H = 84, PAD = 6;
    const t0 = ev.tcaMs - 900000, t1 = ev.tcaMs + 900000;
    const base = missProfile(S.fleet[ev.a], S.fleet[ev.b], t0, t1, 10);
    const withBurn = S.burn && S.burn.dvKmS !== 0 ? missProfile(S.fleet[ev.a], S.fleet[ev.b], t0, t1, 10, S.burn) : null;
    const dMax = Math.max(base.samples.reduce((m, s) => Math.max(m, s.dKm), 0),
      withBurn ? withBurn.samples.reduce((m, s) => Math.max(m, s.dKm), 0) : 0, S.thresholdKm * 2) * 1.05;
    const px = t => PAD + ((t - t0) / (t1 - t0)) * (W - 2 * PAD);
    const py = d => H - PAD - (d / dMax) * (H - 2 * PAD);
    const path = prof => prof.samples.map((s, i) => `${i === 0 ? "M" : "L"}${px(s.tMs).toFixed(1)},${py(s.dKm).toFixed(1)}`).join("");
    const thY = py(S.thresholdKm);
    return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-label="Separation distance around the close approach">
      <line x1="${PAD}" y1="${thY}" x2="${W - PAD}" y2="${thY}" class="pf-th"/>
      <text x="${W - PAD}" y="${thY - 3}" text-anchor="end" class="pf-thlabel">alert ${S.thresholdKm} km</text>
      <path d="${path(base)}" class="pf-base${withBurn ? " pf-base--ghost" : ""}"/>
      ${withBurn ? `<path d="${path(withBurn)}" class="pf-burn"/>` : ""}
      <line x1="${px(ev.tcaMs)}" y1="${PAD}" x2="${px(ev.tcaMs)}" y2="${H - PAD}" class="pf-tca"/>
      <text x="${px(ev.tcaMs) + 3}" y="${PAD + 8}" class="pf-thlabel">closest approach</text>
    </svg>
    <div class="twin__profile-axis"><span>−15 min</span><span>separation (km) vs time</span><span>+15 min</span></div>`;
  }

  // ----- Prediction ------------------------------------------------------------------
  async function runScan() {
    if (S.scanning || !S.n) return;
    S.scanning = true;
    const myScan = ++S.scanId;
    S.predicted = null; S.selected = null;
    drawSelectedOrbits();
    ui.events.innerHTML = "";
    ui.ctas.predict.disabled = true;
    ui.scanbar.hidden = false;
    ui.predictSummary.innerHTML = `The twin is sweeping the next <strong>3 hours</strong> of orbits — every
      object against every other, ~240 future moments…`;
    const t0 = Date.now();
    const horizon = 3 * 3600000;
    const savedOffset = S.twinTimeMs - Date.now();
    const events = await scanAhead({
      fleet: S.fleet, t0Ms: t0, horizonMs: horizon,
      stepSec: 45, screenKm: 30,
      isCancelled: () => S.disposed || myScan !== S.scanId,
      onProgress: (frac, cursorMs, found) => {
        ui.scanFill.style.width = `${(frac * 100).toFixed(1)}%`;
        ui.scanLabel.textContent = `Twin flying ahead… ${fmtOffset(cursorMs - t0)} · ${found} candidate${found === 1 ? "" : "s"}`;
        // The twin globe visibly fast-forwards with the scan cursor.
        S.twinTimeMs = cursorMs;
      },
    });
    if (S.disposed || myScan !== S.scanId) return;
    S.scanning = false;
    ui.scanbar.hidden = true;
    S.twinTimeMs = Date.now() + Math.max(savedOffset, 0);
    if (!events || !events.length) {
      ui.predictSummary.innerHTML = `Quiet sky: no pass closer than 30 km in the next 3 hours among the
        ${S.n.toLocaleString()} objects tracked. Lower the fleet size bias by adding more objects in step 5, or rescan later.`;
      S.predicted = [];
      renderEventList();
      return;
    }
    S.predicted = events;
    const closest = events[0];
    ui.predictSummary.innerHTML = `Done — the twin flew 3 hours of sky in seconds and found
      <strong>${events.length} close pass${events.length === 1 ? "" : "es"}</strong>. The tightest:
      <strong>${fmtDist(closest.dKm)}</strong> at ${fmtUTC(closest.tcaMs, false)} UTC.
      Reality won't know until it gets there.`;
    renderEventList();
    selectEvent(0);
    ui.ctas.predict.disabled = false;
  }

  // ----- Stage machine ----------------------------------------------------------------
  function setStage(stage) {
    const prev = S.stage;
    S.stage = stage;
    root.dataset.stage = stage;
    ui.steps.forEach(b => {
      const active = b.dataset.twStep === stage;
      b.setAttribute("aria-selected", String(active));
      b.classList.toggle("twin__step--done", STAGES.indexOf(b.dataset.twStep) < STAGES.indexOf(stage));
    });
    for (const s of STAGES) ui.cards[s].hidden = s !== stage;
    const splitOn = stage !== "live";
    ui.split.hidden = !splitOn;
    if (splitOn && prev === "live") ui.split.classList.add("twin__split--enter");
    ui.chipClosest.hidden = stage !== "live";
    ui.chipFps.hidden = stage !== "explore";
    if (stage === "live") {
      S.twinSpeed = 1; S.scrubOffsetMin = 0; S.twinTimeMs = Date.now();
      cam.resetView(performance.now());
      S.selected = null; S.burn = null; S.watching = null;
      cam.setChase(null);
      if (S.twinWin) S.twinWin.postHooks.clear();
      drawSelectedOrbits();
    }
    if (stage === "twin") {
      S.twinSpeed = 1;
      ui.scrub.value = String(S.scrubOffsetMin);
      ui.scrubLabel.textContent = fmtOffset(S.scrubOffsetMin * 60000);
    }
    if (stage === "predict" && !S.predicted && !S.scanning) runScan();
    if (stage === "act") {
      // Prefer a tight pass with real lead time — the centimetres-per-second
      // payoff needs an hour or more of drift to land.
      if (S.predicted?.length) {
        const lead = ev => ev.tcaMs - Date.now();
        const current = S.selected;
        if (!current || lead(current) < 60 * 60000) {
          const idx = S.predicted.findIndex(ev => lead(ev) >= 75 * 60000);
          selectEvent(idx >= 0 ? idx : 0);
        } else {
          renderActDetail();
        }
      } else {
        renderActDetail();
      }
      if (S.selected) focusOnEvent(S.selected);
    }
    if (stage === "explore") renderEventList();
  }

  // ----- Wiring ---------------------------------------------------------------------------
  ui.steps.forEach(b => b.addEventListener("click", () => setStage(b.dataset.twStep)));
  ui.ctas.live.addEventListener("click", () => setStage("twin"));
  ui.ctas.twin.addEventListener("click", () => setStage("predict"));
  ui.ctas.predict.addEventListener("click", () => setStage("act"));
  ui.ctas.act.addEventListener("click", () => setStage("explore"));
  ui.restart.addEventListener("click", () => setStage("live"));

  ui.scrub.addEventListener("input", () => {
    S.scrubOffsetMin = +ui.scrub.value;
    S.watching = null; S.twinSpeed = 1;
    S.twinTimeMs = Date.now() + S.scrubOffsetMin * 60000;
    ui.scrubLabel.textContent = fmtOffset(S.scrubOffsetMin * 60000);
  });

  ui.speedBtns.forEach(b => b.addEventListener("click", () => {
    S.twinSpeed = +b.dataset.twSpeed;
    S.watching = null;
    ui.speedBtns.forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  }));
  ui.resync.addEventListener("click", () => {
    S.twinTimeMs = Date.now(); S.twinSpeed = 1; S.watching = null; S.scrubOffsetMin = 0;
    ui.speedBtns.forEach(x => x.setAttribute("aria-pressed", String(x.dataset.twSpeed === "1")));
  });
  ui.rescan.addEventListener("click", runScan);

  ui.threshold.addEventListener("input", () => {
    S.thresholdKm = +ui.threshold.value;
    ui.thresholdLabel.textContent = `${S.thresholdKm.toFixed(1)} km`;
  });
  ui.source.addEventListener("change", () => { S.sourceKey = ui.source.value; loadFleet(); });
  ui.count.addEventListener("change", () => loadFleet());
  ui.noiseToggle.addEventListener("change", () => {
    S.noiseOn = ui.noiseToggle.checked;
    ui.noisectl.hidden = !S.noiseOn;
    S.trackerNoisy.reset();
    ui.twinTag.textContent = S.noiseOn ? "DIGITAL TWIN · NOISY SENSORS" : "DIGITAL TWIN";
  });
  ui.noise.addEventListener("input", () => {
    S.noiseSigmaKm = +ui.noise.value / 1000;
    ui.noiseLabel.textContent = `${ui.noise.value} m`;
  });

  root.addEventListener("click", e => {
    const pick = e.target.closest("[data-tw-ev-pick]");
    if (pick) { selectEvent(+pick.dataset.twEvPick); return; }
    const watch = e.target.closest("[data-tw-ev-watch]");
    if (watch) watchEvent(+watch.dataset.twEvWatch);
  });

  if (small) ui.count.value = "500";

  // ----- Resize ----------------------------------------------------------------------------
  let cssW = 0, cssH = 0;
  function resize() {
    const r = ui.viewport.getBoundingClientRect();
    if (r.width === 0) return;
    cssW = r.width; cssH = r.height;
    sc.renderer.setSize(cssW, cssH, false);
  }
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(ui.viewport);

  // ----- Render helpers ---------------------------------------------------------------------
  const tmpMat = new THREE.Matrix4();
  const tmpPos = new THREE.Vector3();
  const tmpQ = new THREE.Quaternion();
  const tmpS = new THREE.Vector3(1, 1, 1);
  const colNormal = new THREE.Color(0xcfe5dc);
  const colDim    = new THREE.Color(0x6f8797);
  const colAlert  = new THREE.Color(COL.coral);
  const colFamous = new THREE.Color(0xffffff);

  function writeInstances(positions, tracker, nowReal, dimmed) {
    const n = S.n;
    const flag = new Set();
    if (tracker) for (const ev of tracker.active()) if (ev.live) { flag.add(ev.a); flag.add(ev.b); }
    for (let i = 0; i < n; i++) {
      const x = positions[i * 3] * SCENE_SCALE, y = positions[i * 3 + 2] * SCENE_SCALE, z = -positions[i * 3 + 1] * SCENE_SCALE;
      tmpMat.makeTranslation(x, y, z);
      sc.sats.setMatrixAt(i, tmpMat);
      sc.sats.setColorAt(i, flag.has(i) ? colAlert : S.fleet[i].famous ? colFamous : dimmed ? colDim : colNormal);
    }
    sc.sats.instanceMatrix.needsUpdate = true;
    if (sc.sats.instanceColor) sc.sats.instanceColor.needsUpdate = true;

    // Flash lines + pulses for live events
    if (tracker) {
      const evs = tracker.active().filter(e => e.live);
      const m = Math.min(evs.length, 300);
      const fp = sc.flashGeo.attributes.position.array;
      for (let k = 0; k < m; k++) {
        const ev = evs[k];
        fp[k * 6]     = positions[ev.a * 3] * SCENE_SCALE; fp[k * 6 + 1] = positions[ev.a * 3 + 2] * SCENE_SCALE; fp[k * 6 + 2] = -positions[ev.a * 3 + 1] * SCENE_SCALE;
        fp[k * 6 + 3] = positions[ev.b * 3] * SCENE_SCALE; fp[k * 6 + 4] = positions[ev.b * 3 + 2] * SCENE_SCALE; fp[k * 6 + 5] = -positions[ev.b * 3 + 1] * SCENE_SCALE;
      }
      sc.flashGeo.attributes.position.needsUpdate = true;
      sc.flashGeo.setDrawRange(0, m * 2);
      const pulse = reducedMotion ? 1 : 1 + 0.3 * Math.sin(nowReal / 180);
      tmpS.set(pulse, pulse, pulse);
      let pk = 0;
      for (const ev of evs) {
        for (const idx of [ev.a, ev.b]) {
          if (pk >= 600) break;
          tmpPos.set(positions[idx * 3] * SCENE_SCALE, positions[idx * 3 + 2] * SCENE_SCALE, -positions[idx * 3 + 1] * SCENE_SCALE);
          tmpMat.compose(tmpPos, tmpQ, tmpS);
          sc.pulses.setMatrixAt(pk++, tmpMat);
        }
      }
      sc.pulses.count = pk;
      sc.pulses.instanceMatrix.needsUpdate = true;
    } else {
      sc.flashGeo.setDrawRange(0, 0);
      sc.pulses.count = 0;
    }
  }

  function renderPass(vpX, vpW, showTwinGfx) {
    sc.orbitA.line.visible = showTwinGfx && !!S.selected;
    sc.orbitB.line.visible = showTwinGfx && !!S.selected;
    sc.orbitGhost.line.visible = showTwinGfx && !!S.selected && !!S.burn;
    sc.tcaMarker.visible = showTwinGfx && !!S.selected;
    sc.camera.aspect = vpW / Math.max(cssH, 1);
    sc.camera.updateProjectionMatrix();
    sc.renderer.setViewport(vpX, 0, vpW, cssH);
    sc.renderer.setScissor(vpX, 0, vpW, cssH);
    sc.renderer.render(sc.scene, sc.camera);
  }

  // ----- HUD ----------------------------------------------------------------------------------
  let hudTimer = 0, cdTimer = 0, fpsFrames = 0, fpsTime = 0;
  function updateHUD(dt, nowMs) {
    hudTimer += dt;
    fpsFrames++; fpsTime += dt;
    if (hudTimer < 0.25) return;
    hudTimer = 0;
    ui.chipClock.textContent = `${fmtUTC(nowMs)} UTC`;
    ui.clockReal.textContent = `${fmtUTC(nowMs)} UTC`;
    const off = S.twinTimeMs - nowMs;
    ui.clockTwin.textContent = `${fmtUTC(S.twinTimeMs)} UTC · ${fmtOffset(off)}`;
    ui.clockTwin.classList.toggle("twin__split-clock--ahead", Math.abs(off) > 90000);
    if (S.stage === "live" && S.closestNow) {
      const c = S.closestNow;
      ui.chipClosest.hidden = false;
      ui.chipClosest.textContent = `nearest pair right now: ${fmtDist(c.dKm)}`;
      ui.chipClosest.title = `${S.fleet[c.a]?.name} ↔ ${S.fleet[c.b]?.name}`;
    }
    if (S.stage === "explore") {
      ui.chipFps.textContent = `${Math.round(fpsFrames / Math.max(fpsTime, 0.01))} fps`;
      const act = S.tracker.active().filter(e => e.live).length;
      ui.liveCount.textContent = act ? `${act} inside ${S.thresholdKm.toFixed(0)} km now` : "";
      renderNoiseBadges();
    }
    if (fpsTime > 1) { fpsFrames = 0; fpsTime = 0; }
    // Refresh data-age chip every ~30 s
    if (S.fetchedAt && Math.random() < 0.02) updateDataChip();

    cdTimer += 1;
    if (cdTimer >= 4) {  // ≈1 Hz: tick countdowns in place
      cdTimer = 0;
      root.querySelectorAll("[data-tw-cd]").forEach(el => {
        el.textContent = fmtCountdown(+el.dataset.twCd - Date.now());
      });
    }
  }

  // ----- Main loop ------------------------------------------------------------------------------
  let lastT = performance.now();
  let _lastScanFollow = 0;
  function frame(now) {
    if (S.disposed) return;
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - lastT) / 1000);
    lastT = now;
    const nowMs = Date.now();

    cam.tick(dt, now);
    if (sc.tcaMarker.visible) sc.tcaMarker.quaternion.copy(sc.camera.quaternion);

    if (!S.n) {
      sc.earthGroup.rotation.y = gmstAt(nowMs);
      sc.renderer.setScissorTest(false);
      sc.renderer.clear(true, true, true);
      sc.renderer.setViewport(0, 0, cssW, cssH);
      sc.camera.aspect = cssW / Math.max(cssH, 1);
      sc.camera.updateProjectionMatrix();
      sc.renderer.render(sc.scene, sc.camera);
      return;
    }

    // Advance clocks
    if (S.watching && S.twinTimeMs >= S.watching.untilMs) {
      S.twinSpeed = 1; S.watching = null;
      cam.setChase(null);
    }
    if (!S.scanning) S.twinTimeMs += dt * 1000 * S.twinSpeed;
    else /* scan drives twinTimeMs */;
    const realT = nowMs;

    // Propagation windows: reality at 1 s substeps; twin substep scales with
    // speed. While a scan drives the twin clock, follow it at ~2.5 Hz only —
    // re-propagating the fleet every frame would starve the scan itself.
    S.realWin.ensure(realT, 1);
    const twinDt = clamp(S.twinSpeed / 60, 1, 10);
    if (!S.scanning) {
      S.twinWin.ensure(S.twinTimeMs, twinDt);
    } else if (now - _lastScanFollow > 400) {
      _lastScanFollow = now;
      S.twinWin.ensure(S.twinTimeMs, twinDt);
    }

    sc.renderer.setScissorTest(true);
    sc.renderer.clear(true, true, true);

    if (S.stage === "live") {
      sc.earthGroup.rotation.y = gmstAt(nowMs);
      S.realWin.sample(realT, S.renderPos);
      writeInstances(S.renderPos, null, now, false);
      renderPass(0, cssW, false);
      placeLabels(S.renderPos, 0, cssW, cssW, cssH);
    } else {
      const half = Math.floor(cssW / 2);
      // Left: reality (dimmed slightly, no twin graphics), Earth at real GMST
      sc.earthGroup.rotation.y = gmstAt(nowMs);
      S.realWin.sample(realT, S.renderPos);
      writeInstances(S.renderPos, null, now, true);
      renderPass(0, half, false);
      placeLabels(S.renderPos, 0, half, cssW, cssH);
      // Right: the twin, Earth rotated to the twin's clock
      sc.earthGroup.rotation.y = gmstAt(S.twinTimeMs);
      const useNoisy = S.noiseOn && S.stage === "explore";
      S.twinWin.sample(S.twinTimeMs, S.renderPos);
      let shown = S.renderPos;
      if (useNoisy) { S.noise.apply(S.renderPos, S.renderPosNoisy); shown = S.renderPosNoisy; }
      writeInstances(shown, useNoisy ? S.trackerNoisy : S.tracker, now, false);
      renderPass(half, cssW - half, true);
    }
    sc.renderer.setScissorTest(false);

    updateHUD(dt, nowMs);
  }

  // ----- Boot -------------------------------------------------------------------------------------
  setStage("live");
  setLoading("Contacting CelesTrak for live orbits…");
  loadFleet();
  requestAnimationFrame(frame);

  return function dispose() {
    S.disposed = true;
    ro.disconnect();
    sc.renderer.dispose();
  };
}

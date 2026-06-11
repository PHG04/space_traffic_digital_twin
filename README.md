# Space Traffic Digital Twin

A digital twin of low Earth orbit: a software copy of the real satellite population,
synchronised with public tracking data, that can run ahead of the wall clock to find
close approaches before they happen — and rehearse the avoidance manoeuvre that fixes
them, without touching the real sky.

The repo contains **two implementations sharing one architecture**:

| | Where | Stack | Role |
|---|---|---|---|
| **Browser twin** | [`web/`](web/) | satellite.js (SGP4) + Three.js, no build step | The demo. Live data, guided tour, runs anywhere. |
| **Research sim** | [`src/`](src/) + [`dashboard/`](dashboard/) | Python, NumPy/SciPy, Poliastro, Dash | The original prototype the twin grew from. |

**▶ Try it: [pierregathy.com/projects/space-traffic-digital-twin](https://pierregathy.com/projects/space-traffic-digital-twin.html)**

## What the demo teaches

The guided tour walks the digital-twin loop in four beats:

1. **Mirror** — the globe is a live map of the real catalog (CelesTrak GP elements,
   propagated with SGP4 in your browser, Earth rotated to true GMST). The ISS is tagged
   so you have a landmark. Reality plays at 1× — it has no other setting.
2. **Predict** — the view splits: reality on the left, the twin on the right. The twin
   sweeps the next 3 hours of orbits in about a second and ranks the closest approaches
   it finds, with countdowns. You can replay any of them at 25× before they happen.
3. **Act** — pick the tightest pass and nudge one satellite along-track by centimetres
   per second. The Clohessy–Wiltshire drift (≈ −3·Δv per second of lead time) opens the
   miss distance by kilometres. The real satellite never moved.
4. **Explore** — free sandbox: playback speed, alert threshold, constellation choice,
   plus a researcher mode that feeds the twin time-correlated sensor noise and tallies
   the close passes it consequently misses or hallucinates — the project's original
   research question.

## Running the browser twin locally

```bash
cd web
python -m http.server 8000   # any static server works; file:// won't (ES modules)
# open http://localhost:8000
```

Both the hosted and the standalone page fetch live elements through a small
caching proxy (`api/tle.js` in the website repo, CORS-open) because CelesTrak
doesn't serve CORS headers itself. If the proxy is unreachable, the page falls
back to the bundled snapshot in `web/data/` (stamped in `tle-manifest.json`) —
and the corner badge labels the data age either way.

### Mirroring contract

The browser twin is developed here and mirrored byte-for-byte into the website repo
([PHG04/website](https://github.com/PHG04/website)) — edit in one place, copy to the other:

| canonical (this repo) | mirror (website repo) |
|---|---|
| `web/js/digital-twin.js` | `assets/js/digital-twin.js` |
| `web/css/digital-twin.css` | `assets/css/digital-twin.css` |
| `web/data/*` | `assets/data/*` |

## Architecture

One tick loop, six concerns. Each Python module has a matching section in
`web/js/digital-twin.js` (marked `§2–§9` in the file header):

| Python | Browser twin | Job |
|---|---|---|
| `src/orbital_mechanics/` | §3 | Kepler + J2 propagation (Python) / SGP4 on real TLEs (JS), positions at fixed substeps, interpolated for rendering |
| `src/conjunction_detection/spatial_index.py` | §4 | KDTree range queries (Python) / uniform spatial hash (JS) |
| `src/conjunction_detection/conjunction_analyzer.py` | §5 | Pair screening; the JS adds an analytic closest-approach solve between substeps so fast crossings can't slip between frames |
| `src/risk_assessment/` | §6 | Collision probability from miss distance + lead-time-dependent position uncertainty |
| `src/sensor_simulation/` | §7 | Gaussian / correlated tracking noise (Ornstein–Uhlenbeck in the JS) |
| — | §8 | What-if manoeuvres via Clohessy–Wiltshire offsets (twin-side only) |
| `dashboard/app.py` (Dash) | §10+ | 3D visualisation, event list, controls |

## Running the Python sim

```bash
pip install -r requirements.txt
python demo.py                 # console walkthrough on synthetic satellites, offline
python dashboard/app.py        # Dash dashboard at http://localhost:8050
```

The dashboard can enrich the view with live positions for a handful of well-known
satellites via the N2YO API — set `N2YO_API_KEY` in your environment (see
`.env.example`, free tier at n2yo.com). Without a key it falls back to simulated
positions. Configuration knobs (thresholds, fleet size, time step) live in
`config/config.yaml`.

## Data sources

- **CelesTrak GP elements** (https://celestrak.org) — the public two-line element
  catalog; `web/data/` holds a filtered LEO snapshot (11–17 rev/day) for offline use.
- **SGP4** via [satellite.js](https://github.com/shashwatak/satellite-js) — the
  standard propagator for exactly this data.
- **Natural Earth 110m** coastlines (public domain), drawn onto the globe texture.

## Honest limitations

This is a working prototype built to demonstrate the architecture, not an
operational screening system:

- Screening uses point estimates; an operational system propagates full covariance
  and computes Pc from the joint uncertainty ellipsoid (Foster, Chan, Alfano…).
  The probabilities shown are order-of-magnitude indicators on a deliberately
  conservative ~km-scale sigma, which is honest for public GP data.
- The browser tracks a subset of the catalog (up to 2,500 of ~15k LEO objects) to
  stay at 60 fps on ordinary hardware; the subset is an even stride, famous
  spacecraft pinned.
- GP elements are themselves km-accurate and degrade over days; the demo shows the
  data age rather than pretending otherwise.
- The Python sim's propagators are Kepler/J2 on osculating elements — fine for
  conjunction-rate studies on synthetic fleets, not for flying real TLEs (that's
  what SGP4 is for, which is why the browser twin uses it).

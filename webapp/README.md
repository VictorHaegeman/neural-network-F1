# F1 Neural Betting Guide Webapp

This folder is the local simulation interface built from
`webapp/data/betting_guide_data.json`.

## Pages

- `index.html`: scoreboard for all upcoming races, with quick model signals.
- `race.html?race=<race_id>`: GP dossier with circuit image, weather, circuit
  history, strategy notes, market filters and simulated ticket.
- `driver.html?race=<race_id>&driver=<driver_code>`: driver file with model
  probability, fair odds, circuit history and nearby market comparison.

## Build Data

Run this from the repo root after regenerating predictions:

```powershell
.\.venv\Scripts\python.exe scripts\build_betting_guide_data.py
```

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m http.server 8000
```

Then open:

```text
http://127.0.0.1:8000/webapp/
```

The betting flow is simulation-only. Real-money betting would need licensing,
identity checks, payment security and responsible-gambling controls.

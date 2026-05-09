# Neural Network Betting Guide - Work Plan

## Verdict

The idea is interesting if it is framed as a prediction and bankroll-simulation
tool first. The neural network becomes the centre of the product: it produces a
probability for each driver, the interface turns those probabilities into fair
odds and risk levels, and the user can test betting strategies before ever
placing a real-world bet.

The real-money version is a different project. A site that accepts stakes and
redistributes a pool is gambling infrastructure, so it would need licensing,
payment security, identity checks, fraud controls, responsible-gambling limits
and local legal review. This repo should stay in simulation mode until the ML
quality and product value are proven.

## Product Direction

Build an F1 prediction cockpit:

- neural-network predictions are the primary source of truth
- each race has a ranked grid, top-10 probability, fair decimal odds and risk tag
- the user can enter offered odds to calculate expected value
- a virtual betting slip tracks bankroll, stake size and exposure
- a simulated pari-mutuel pool redistributes fictional money after race results
- completed races can later be used to compare predicted value with real results

## Step-by-Step Execution

### Step 1 - Make the NN output product-ready

Status: executed in this MVP through `scripts/build_betting_guide_data.py`.

Tasks:

- read `outputs/predictions/neural_network/upcoming_top10_predictions.csv`
- group predictions by race
- calculate fair odds as `1 / top10_probability`
- calculate the bubble gap between ranks 10 and 11
- attach neural-network summary metrics
- export a clean JSON payload for the web interface

Validation:

```powershell
python scripts/build_betting_guide_data.py
```

### Step 2 - Build the guide interface

Status: executed in this MVP through `webapp/index.html`, `webapp/styles.css`
and `webapp/app.js`.

Tasks:

- race selector
- bankroll and stake controls
- prediction table with driver headshots
- fair odds and manual offered-odds input
- expected-value calculation
- virtual betting slip
- simulated pari-mutuel settlement panel
- race dossier for the selected Grand Prix:
  - forecast/weather context
  - circuit chaos and safety-car risk
  - recent winners and race conditions
  - strongest drivers at the circuit
  - driver-level "performance vs luck/chaos" notes
  - expected stop/compound strategy
  - team tyre tendencies

Validation:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/webapp/
```

### Step 3 - Improve the neural network

Status: partially executed. The general `neural_network_mlp` classifier now uses
the best known tuned architecture from `outputs/neural_network_summary.json`:
`(80, 40)`, `tanh`, `alpha=0.002`, `learning_rate_init=0.0007`.

Tasks:

- keep the 2025 temporal holdout as the main validation gate
- add calibration metrics, especially Brier score and calibration curves
- tune for ranking quality, not only classification score
- compare the dedicated NN with random forest and histogram boosting per race
- add ensemble/stacked probabilities only if they beat the NN consistently
- retrain on full historical data only after the holdout choice is locked

Validation:

```powershell
python scripts/tune_neural_network.py --force
python scripts/evaluate_models.py
```

### Step 4 - Add betting-grade evaluation

Status: next product-quality milestone.

Tasks:

- store historical odds when available
- backtest flat staking, fractional Kelly and top-N portfolio strategies
- report ROI, drawdown, hit rate, average edge and closing-line movement
- separate "model is accurate" from "bet is valuable"

Validation:

```powershell
python scripts/backtest_betting_strategy.py
```

### Step 5 - Add race-result settlement

Status: next interface milestone.

Tasks:

- import completed race results after the race
- mark winning top-10 drivers automatically
- settle the simulated pool
- persist historical virtual tickets
- show prediction error and profit/loss by race

### Step 6 - Real-money path, only if needed

Status: out of scope for this repo right now.

Tasks:

- legal review by target country
- age and identity checks
- payment provider integration
- anti-fraud monitoring
- responsible-gambling limits
- audit logs and dispute handling
- terms, privacy policy and risk warnings

## MVP Deliverables Added

- `docs/NEURAL_BETTING_GUIDE_PLAN.md`
- `scripts/build_betting_guide_data.py`
- `webapp/index.html`
- `webapp/race.html`
- `webapp/styles.css`
- `webapp/app.js`
- `webapp/race.js`
- `webapp/assets/circuits/*.svg`
- `webapp/data/betting_guide_data.json`

## Interface Structure

- `webapp/index.html` is the scoreboard: quick race selector, upcoming GP cards
  and the model ranking table.
- `webapp/race.html?race=<race_id>` is the GP dossier: circuit image, weather,
  circuit history, strategy, driver detail and the simulated betting slip.

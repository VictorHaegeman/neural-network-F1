# Data Coverage and Import Recommendations

Last checked: 2026-05-07.

## Current Data Coverage

The live Jolpica audit for the 2026 season shows that no completed race is
currently missing from the local project data.

| Item | Current coverage |
|---|---:|
| 2026 scheduled races | 22 |
| 2026 races with API results | 4 |
| 2026 races with local results | 4 |
| 2026 races in final dataset | 4 |
| Final dataset rows | 6,999 |
| Final dataset columns | 198 |
| Final dataset races | 333 |
| Final dataset seasons | 2010-2026 |
| Missing values in final dataset | 0 |

The next unavailable 2026 race in the current schedule is `2026_05` Canadian
Grand Prix, dated 2026-05-24. It should be imported only after Jolpica exposes
race results and qualifying.

## Raw Feature Coverage

| Source/table | Coverage |
|---|---:|
| `race_results.csv` | 6,999 rows, 333 races |
| `qualifying_results.csv` | 6,976 rows, 333 races |
| `pit_stop_events.csv` | 12,294 rows, 313 races |
| `weather_data.csv` | 333 races |
| `fastf1_weather.csv` | 177 races |
| `fastf1_lap_summaries.csv` | 100 races |
| `fastf1_driver_form.csv` | 6,999 rows, 333 races |
| `fastf1_race_control.csv` | 177 races |
| `race_control_history.csv` | 333 races |

Pit-stop gaps are mostly expected: Jolpica does not expose detailed pit-stop
rows for the 2010 season, and the 2021 Belgian Grand Prix has no standard race
pit-stop signal to import.

## Feature Families Already Present

| Family | Columns in final dataset |
|---|---:|
| Form and standings | 73 |
| Race-control and safety-car history | 26 |
| Weather | 24 |
| FastF1 lap/tyre proxies | 9 |
| Pit strategy | 7 |
| Qualifying/grid | 5 |

## Quick Ablation Check

The current champion classifier is histogram gradient boosting. A quick
holdout ablation on the 2025 season gives the following race precision@10:

| Feature setup | Race precision@10 |
|---|---:|
| All features | 0.779 |
| Without pit strategy | 0.783 |
| Without qualifying/grid | 0.779 |
| Without FastF1 lap/tyre | 0.779 |
| Without weather | 0.771 |
| Without race-control history | 0.771 |

This suggests that weather and race-control history are currently the most
useful import areas. Pit-stop features may need better engineering before they
improve the model reliably.

## Best Next Imports

1. **Future completed 2026 races**

   Import race results, qualifying and pit stops after each Grand Prix becomes
   available in Jolpica.

   ```powershell
   python scripts/audit_data_coverage.py --season 2026
   python scripts/import_missing_completed_races.py --season 2026
   python scripts/generate_final_dataset.py
   python scripts/train_model.py --model hist_gradient_boosting
   ```

2. **Remaining FastF1 lap and tyre summaries**

   The largest real feature gap is FastF1 lap/tyre coverage for 2020-2023.
   These features are useful for modelling driver pace, stint behaviour and wet
   tyre usage, but the timing API is rate limited. It is best to rerun this
   incrementally rather than forcing a complete refresh.

   ```powershell
   python scripts/generate_fastf1_features.py --start-year 2020 --end-year 2023 --incremental
   ```

3. **Richer weather forecast/pre-race features**

   The current dataset has historical race-day weather. For upcoming races, the
   model would benefit from explicit pre-race forecast snapshots, for example
   predicted rain probability, forecast temperature, wind and humidity at race
   start. This should be stored separately from actual post-race weather to
   avoid leakage.

4. **Better pit-stop strategy features**

   Existing pit-stop event data is present, but the current features are simple
   previous-race aggregates. Better features could include circuit-level average
   stops, pit-lane loss, tyre-degradation proxy, undercut/overcut frequency and
   team-specific stop consistency.

5. **Driver/team market and reliability signals**

   External data such as driver changes, power-unit penalties, grid penalties,
   team upgrades or betting odds could improve precision, but these sources
   need careful citation and leakage control.

## Recommendation

For the assignment, the dataset is already complete enough and no immediate
race-result import is missing. The best precision-improvement work would be:

1. keep importing each new completed 2026 race;
2. fill FastF1 lap/tyre gaps for 2020-2023 when API limits allow it;
3. add clean pre-race forecast features for upcoming predictions;
4. re-engineer pit-stop strategy features before relying on them more heavily.

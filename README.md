# PVision — Solar PV Power Prediction from Weather Conditions

Estimating and forecasting the power output of a rooftop solar installation from weather
conditions alone, so a site with no production meter can still know what it is generating.

Capstone project, Team 9 — Lina Baageel, Maryam, Jawaher.

## The problem

Many small PV installations record weather but never meter their own production. Without a
production reading, an owner cannot tell an underperforming array from a cloudy week. PVision
acts as a virtual meter: it estimates output from weather now, and forecasts it up to two hours
ahead.

## Data

`solar_weather.csv` — 196,776 records at 15-minute intervals, January 2017 to August 2022, from a
single rooftop system of roughly 20 kW peak in northern Europe (about 52°N).

- The target `Energy delta[Wh]` is converted to `Power[kW]` (`× 4 / 1000`).
- 51.3% of records are night-time zeros; after filtering to daylight, 102,316 records remain.
- Irradiance (GHI) correlates 0.91 with output. Output swings roughly tenfold between summer and
  winter (June mean 3.71 kW, December 0.36 kW).

**Units to be aware of.** GHI in this dataset is energy per 15-minute step (Wh/m², max ≈ 229), not
instantaneous W/m². `dayLength` is in minutes (450–1020). Timestamps are UTC. Wind speed is m/s and
snowfall is mm. Every live weather call in `Pipeline/` converts the API's units to match.

## Split

Strictly chronological — a random split would leak future patterns into training.

| Split | Period | Rows |
|---|---|---|
| Train | 2017–2020 | 71,647 |
| Validation | 2021 | 17,735 |
| Test | Jan–Aug 2022 | 12,934 |

The test year was opened once, at the end.

## Models

Thirteen weather-only features: `GHI, temp, pressure, humidity, wind_speed, rain_1h, snow_1h,
clouds_all, dayLength, hour_sin, hour_cos, month_sin, month_cos`.

- **Nowcasting** — one Random Forest estimating output at the current moment.
- **Forecasting** — eight Random Forests, one per horizon from 15 minutes to 2 hours, each
  predicting output that far ahead from the present weather.
- **Baseline** — persistence (assume output stays at its current value).

All forests use `n_estimators=100`, `min_samples_leaf=10`, `random_state=42`.

### Why no lagged production features

Lagged output (`power_lag1`, …) improves the forecast, but requires live telemetry from the plant,
which the application cannot access. We measured what excluding it costs: 0.044 R² at the 15-minute
horizon, shrinking to 0.014 at two hours. The deployed models therefore use weather only.

## Results

*(Verify these against the current notebook outputs before submitting.)*

**Nowcasting**

| Split | MAE (kW) | R² |
|---|---|---|
| Validation 2021 | 0.966 | 0.869 |
| Test 2022 | 1.021 | 0.859 |

**Forecasting, test 2022** — R² falls from 0.831 at 15 minutes to 0.684 at two hours, while the
persistence baseline collapses from 0.852 to 0.130 over the same range. The value of the model
grows with the horizon.

## Repository layout

```
Data/         raw and split datasets
EDA/          initial exploration
Models/       training and evaluation notebooks, saved models
Pipeline/     live prediction scripts (call the weather API)
app/          Streamlit interface
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app/PVapp.py
```

The app covers five cities (Warsaw, Berlin, Amsterdam, London, Hamburg). Choosing **Now** uses the
nowcasting model, **15min–2hour** use the forecasting models.

Predictions are explained in plain language by an LLM (Llama 3.1 via Hugging Face). This is
optional: set `HF_TOKEN` in `app/.env`, and without it the app falls back to a rule-based
explanation. The number itself always comes from the Random Forest, never from the LLM.

To retrain, run `Models/Comparing_models_nowcasting.ipynb` and
`Models/comparing_models_forecasting.ipynb`, then `Models/final_test_evaluation.ipynb`.

## Limitations

- One installation, one climate. Predictions for other cities assume a comparable system at a
  similar latitude.
- The dataset ends in August 2022, so the test year has no autumn or winter.
- The GHI unit conversion is inferred from the data's range rather than documented at source.
- Forecasts for a chosen day inherit the weather forecast's own error on top of the model's.
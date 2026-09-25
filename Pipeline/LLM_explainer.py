"""Explain a PV power prediction in plain language, using an LLM with a rule-based fallback."""

import os
import time
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The token lives in app/.env, which is not committed
load_dotenv(os.path.join(BASE_DIR, '..', 'app', '.env'))
load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']


# Turn each reading into the plain word that describes it, so the explanation
# never has to judge the numbers itself and cannot contradict them
def describe_conditions(row):
    irradiance = row["GHI"] * 4
    clouds = row["clouds_all"]
    humidity = row["humidity"]

    if irradiance < 50:
        irradiance_word = "very low"
    elif irradiance < 200:
        irradiance_word = "low"
    elif irradiance < 400:
        irradiance_word = "moderate"
    elif irradiance < 650:
        irradiance_word = "high"
    else:
        irradiance_word = "very high"

    if clouds < 20:
        cloud_word = "clear sky"
    elif clouds < 50:
        cloud_word = "partly cloudy"
    elif clouds < 85:
        cloud_word = "mostly cloudy"
    else:
        cloud_word = "overcast"

    if humidity < 40:
        humidity_word = "low"
    elif humidity < 65:
        humidity_word = "moderate"
    else:
        humidity_word = "high"

    return irradiance, irradiance_word, cloud_word, humidity_word


# A short explanation built from the readings alone, used when the LLM is unavailable
def get_rule_based_explanation(is_daylight, irradiance, clouds_all, prediction):
    if not is_daylight:
        return ("The sun is below the horizon, so there is no solar irradiance reaching the "
                "panels and the expected output is zero.")
    if irradiance < 50:
        return (f"Irradiance is only {irradiance:.0f} W/m² with {clouds_all:.0f}% cloud cover, "
                "so very little sunlight is reaching the panels — expect minimal output.")
    if prediction < 5:
        return (f"With {irradiance:.0f} W/m² of irradiance and {clouds_all:.0f}% cloud cover, "
                "conditions support a moderate output — sunlight is present but partly blocked or weak.")
    return (f"Strong irradiance ({irradiance:.0f} W/m²) and {clouds_all:.0f}% cloud cover "
            "make this a high-output window — close to ideal conditions for solar generation.")


def build_prompt(city_name, prediction, row, target_time, is_daylight):
    irradiance, irradiance_word, cloud_word, humidity_word = describe_conditions(row)
    situation = "It is daytime" if is_daylight else "It is nighttime, no solar irradiance"

    return f"""You are a professional solar energy analyst explaining a solar PV power prediction to a general audience.

Your task is to explain whether the predicted solar power output is reasonable based on the weather and time-related features used by the prediction model.

Prediction context:
- City: {city_name}
- Predicted power output: {prediction:.2f} kW

Weather and environmental inputs. The word in brackets is the correct description of that
value — use it, and never describe the value in any other way:
- Solar irradiance: {irradiance:.0f} W/m² [{irradiance_word}]
- Cloud cover: {row['clouds_all']} % [{cloud_word}]
- Relative humidity: {row['humidity']} % [{humidity_word}]
- Temperature: {row['temp']} °C
- Pressure: {row['pressure']} hPa
- Wind speed: {row['wind_speed']} m/s
- Rain: {row['rain_1h']} mm
- Snowfall: {row['snow_1h']} mm

Time-related inputs (local time at the specified city):
- Time of day: {target_time.strftime('%I:%M %p')}
- Month: {MONTH_NAMES[target_time.month - 1]}
- Daytime condition: {situation}

Analysis requirements:
1. Assess whether the predicted power output is reasonable given the provided conditions.
2. Identify the most important factors affecting the prediction, especially solar radiation, cloud cover, daylight, and time of day.
3. Mention other weather factors only when they meaningfully contribute to the explanation.
4. If the conditions indicate nighttime or very low solar radiation, clearly explain why the expected solar power should be very low or zero.
5. Do not invent weather conditions, values, or relationships that are not supported by the provided data.
6. Keep the explanation concise, clear, and understandable to a non-technical user.
7. Give a 2-3 sentence explanation only.
8. Describe every value exactly as its bracketed word says. Never call the sky clear when it is
   marked mostly cloudy or overcast, and never call humidity low when it is marked high.
9. Do not write the bracketed words in square brackets; work them into the sentence naturally.
10. When irradiance is high but the sky is overcast, say that the cloud is thin enough to let
    strong sunlight through, rather than treating the two as contradictory.
"""


# Returns (explanation, came_from_llm, reason_it_fell_back)
def get_llm_explanation(city_name, prediction, row, target_time, is_daylight):
    irradiance, _, _, _ = describe_conditions(row)
    fallback = get_rule_based_explanation(is_daylight, irradiance, row["clouds_all"], prediction)

    token = os.environ.get("HF_TOKEN")
    if not token:
        return fallback, False, "No HF_TOKEN found in .env"

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": build_prompt(city_name, prediction, row, target_time, is_daylight)}],
        "max_tokens": 400,
        "temperature": 0.3,
    }

    # The model is sometimes still loading, so a 503 is worth retrying
    last_problem = None
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        except requests.exceptions.RequestException as error:
            last_problem = f"Request failed: {error}"
            break

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            return content.strip(), True, None

        if response.status_code == 503:
            last_problem = "Model was still loading after three attempts"
            time.sleep(10)
            continue

        last_problem = f"HTTP {response.status_code}: {response.text[:200]}"
        break

    return fallback, False, last_problem
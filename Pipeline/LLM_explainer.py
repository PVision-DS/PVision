import requests
import os
from dotenv import load_dotenv
import time


load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

API_URL = "https://router.huggingface.co/v1/chat/completions"

def explain_prediction(city_name, prediction, weather, ghi_adjusted, day_length, now, situation, horizon_label='now'):

    if weather.get('is_day', 1) == 0 or weather['shortwave_radiation'] /4 <= 0.1:
        situation = "It is nighttime, no solar irradiance"
    else:
        situation = "It is daytime"



    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

    prompt = f"""You are a professional solar energy analyst explaining a solar PV power prediction to a general audience.

Your task is to explain whether the predicted solar power output is reasonable based on the weather and time-related features used by the prediction model.

Prediction context:
- City: {city_name}
- Forecast horizon: {horizon_label}
- Predicted power output: {prediction:.2f} kW

Weather and environmental inputs:
- Shortwave radiation / adjusted GHI: {ghi_adjusted:.2f} W/m²
- Temperature: {weather.get('temperature_2m', 'N/A')} °C
- Surface pressure: {weather.get('surface_pressure', 'N/A')} hPa
- Relative humidity: {weather.get('relative_humidity_2m', 'N/A')} %
- Wind speed: {weather.get('wind_speed_10m', 'N/A')} km/h
- Rain: {weather.get('rain', 0)} mm
- Snowfall: {weather.get('snowfall', 0)} cm
- Cloud cover: {weather.get('cloud_cover', 'N/A')} %
- Estimated daylight duration: {day_length:.2f} hours

Time-related inputs (local time at the specified city):
- Time of day: {now.strftime('%I:%M %p')}
- Month: {month_names[now.month - 1]}
- Daytime condition: {situation}

Analysis requirements:
1. Assess whether the predicted power output is reasonable given the provided conditions.
2. Identify the most important factors affecting the prediction, especially solar radiation, cloud cover, daylight, and time of day.
3. Mention other weather factors such as temperature, humidity, rain, snowfall, wind, or pressure only when they meaningfully contribute to the explanation.
4. If the conditions indicate nighttime or very low solar radiation, clearly explain why the expected solar power should be very low or zero.
5. Do not invent weather conditions, values, or relationships that are not supported by the provided data.
6. Keep the explanation concise, clear, and understandable to a non-technical user.
7. Give a 2-3 sentence explanation only.
"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 400,
        "temperature": 0.7
    }


    for attempt in range(3):

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
           result = response.json()
           print('STATUS CODE:', response.status_code)
           content = result["choices"][0]["message"]["content"]
           print('CONTENT LENGTH:', len(content))
           return content.strip()

        elif response.status_code == 503:
            time.sleep(10)

        else:
            print(response.text)
            return f"Could not generate explanation (error: {response.status_code})"

    return "Could not generate explanation after multiple attempts"
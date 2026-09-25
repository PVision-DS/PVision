import joblib
import numpy as np
import requests
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'nowcast_model.pkl')

CITIES = {
    'Warsaw': (52.2297, 21.0122),
    'Berlin': (52.5200, 13.4050),
    'Amsterdam': (52.3676, 4.9041),
    'London': (51.5074, -0.1278),
    'Hamburg': (53.5511, 9.9937),
}

# Fetch the current weather for a city, in the units the training data uses
def get_current_weather(lat, lon):
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat,
        'longitude': lon,
        'current': 'temperature_2m,relative_humidity_2m,cloud_cover,'
                   'wind_speed_10m,shortwave_radiation,pressure_msl,rain,snowfall,is_day',
        'wind_speed_unit': 'ms',
        'timezone': 'UTC'
    }
    return requests.get(url, params=params).json()['current']

def calculate_day_length(lat, date):
    day_of_year = date.timetuple().tm_yday
    lat_rad = np.radians(lat)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    cos_hour_angle = -np.tan(lat_rad) * np.tan(decl)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    hour_angle = np.arccos(cos_hour_angle)
    day_length_hours = (2 * hour_angle * 24) / (2 * np.pi)
    return day_length_hours * 60

# Estimate the power being produced right now
def predict_nowcast(city_name):
    city_name = city_name.strip().title()
    if city_name not in CITIES:
        print('Error: city not supported:', city_name)
        return None

    lat, lon = CITIES[city_name]
    weather = get_current_weather(lat, lon)
    now = datetime.fromisoformat(weather['time'])

    ghi_adjusted = weather['shortwave_radiation'] / 4   # API returns W/m2, training data is Wh/m2 per 15-min step

    if weather.get('is_day', 1) == 0 or ghi_adjusted <= 0.1:
        print("It's nighttime — no solar output expected")
        return 0.0

    day_length = calculate_day_length(lat, now)

    features = [[
        ghi_adjusted,
        weather['temperature_2m'],
        weather['pressure_msl'],
        weather['relative_humidity_2m'],
        weather['wind_speed_10m'],
        weather.get('rain', 0),
        weather.get('snowfall', 0) * 10,   # API returns cm, training data is mm
        weather['cloud_cover'],
        day_length,
        np.sin(2 * np.pi * now.hour / 24), np.cos(2 * np.pi * now.hour / 24),
        np.sin(2 * np.pi * now.month / 12), np.cos(2 * np.pi * now.month / 12)
    ]]

    model = joblib.load(MODEL_PATH)
    return model.predict(features)[0]
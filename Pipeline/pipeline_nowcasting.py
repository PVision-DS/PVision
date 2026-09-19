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

def get_current_weather(lat, lon):
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,cloud_cover,"
           f"wind_speed_10m,shortwave_radiation,surface_pressure,rain,snowfall,is_day"
           f"&timezone=auto")
    return requests.get(url).json()['current']

def calculate_day_length(lat, date):
    day_of_year = date.timetuple().tm_yday
    lat_rad = np.radians(lat)
    decl = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    cos_hour_angle = -np.tan(lat_rad) * np.tan(decl)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    hour_angle = np.arccos(cos_hour_angle)
    day_length_hours = (2 * hour_angle * 24) / (2 * np.pi)
    return day_length_hours * 60   

def predict_nowcast(city_name):
    lat, lon = CITIES[city_name]
    weather = get_current_weather(lat, lon)
    now = datetime.fromisoformat(weather['time'])   

    ghi_adjusted = weather['shortwave_radiation'] / 4

    if weather.get('is_day', 1) == 0 or ghi_adjusted <= 0.1:
        print("It's nighttime — no solar output expected")
        return 0.0

    day_length = calculate_day_length(lat, now)

    features = [[
        ghi_adjusted,
        weather['temperature_2m'],
        weather['surface_pressure'],
        weather['relative_humidity_2m'],
        weather['wind_speed_10m'],
        weather.get('rain', 0),
        weather.get('snowfall', 0),
        weather['cloud_cover'],
        day_length,
        np.sin(2*np.pi*now.hour/24), np.cos(2*np.pi*now.hour/24),
        np.sin(2*np.pi*now.month/12), np.cos(2*np.pi*now.month/12)
    ]]

    model = joblib.load(MODEL_PATH)
    return model.predict(features)[0]
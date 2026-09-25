"""Forecasting pipeline: fetch weather for a supported city and predict PV output."""

import os
import numpy as np
import pandas as pd
import requests
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Models', 'models')

# The 13 features the deployed models use, all available from the weather API
features_weather_only = ['GHI', 'temp', 'pressure', 'humidity', 'wind_speed',
                          'rain_1h', 'snow_1h', 'clouds_all', 'dayLength',
                          'hour_sin', 'hour_cos', 'month_sin', 'month_cos']

# Horizons the saved models cover, with how far ahead each one looks in minutes
horizon_minutes = {'15min': 15, '30min': 30, '45min': 45, '1hour': 60,
                    '75min': 75, '90min': 90, '105min': 105, '2hour': 120}
allowed_horizons = list(horizon_minutes)

# Cities supported by this pipeline, mapped to (lat, lon)
allowed_cities = {'Warsaw': (52.23, 21.01), 'Berlin': (52.52, 13.40), 'Amsterdam': (52.37, 4.90),
                   'London': (51.51, -0.13), 'Hamburg': (53.55, 9.99)}

WEATHER_VARIABLES = ('temperature_2m,relative_humidity_2m,pressure_msl,cloud_cover,'
                     'wind_speed_10m,precipitation,snowfall,shortwave_radiation,is_day')


# Turn one row of the API response into the units the models were trained on
def build_weather(block, index):
    ghi_api = block['shortwave_radiation'][index]
    if ghi_api is None:
        return None

    return {
        'temp': block['temperature_2m'][index],
        'humidity': block['relative_humidity_2m'][index],
        'pressure': block['pressure_msl'][index],
        'clouds_all': block['cloud_cover'][index],
        'wind_speed': block['wind_speed_10m'][index],
        'rain_1h': block['precipitation'][index],
        'snow_1h': block['snowfall'][index] * 10,   # API returns cm, training data is mm
        'GHI': ghi_api / 4,                          # API returns W/m2, training data is Wh/m2 per 15-min step
        'is_day': block['is_day'][index]
    }


# Fetch weather for a supported city at a specific UTC time.
# The training data is 15-minute, so 15-minute weather is used when the API has it,
# and the hourly series is the fallback for times beyond its range.
def get_weather_at(city_name, current_datetime):
    city_name = city_name.strip().title()
    if city_name not in allowed_cities:
        print('Error: city not found in allowed_cities:', city_name)
        return None

    lat, lon = allowed_cities[city_name]
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': lat,
        'longitude': lon,
        'minutely_15': WEATHER_VARIABLES,
        'hourly': WEATHER_VARIABLES,
        'wind_speed_unit': 'ms',
        'timezone': 'UTC',
        'forecast_days': 7
    }
    response = requests.get(url, params=params)
    data = response.json()

    quarter_hour = current_datetime.strftime('%Y-%m-%dT%H:%M')
    minutely = data.get('minutely_15', {})
    if quarter_hour in minutely.get('time', []):
        weather = build_weather(minutely, minutely['time'].index(quarter_hour))
        if weather is not None:
            return weather

    # Fall back to the hourly series, rounding down to the hour
    whole_hour = current_datetime.strftime('%Y-%m-%dT%H:00')
    hourly = data.get('hourly', {})
    if whole_hour not in hourly.get('time', []):
        print('Error: requested time is outside the API window:', quarter_hour)
        return None

    weather = build_weather(hourly, hourly['time'].index(whole_hour))
    if weather is None:
        print('Error: GHI (shortwave_radiation) is missing for', city_name)
    return weather


# Approximate day length and cyclical hour/month features for a given datetime
def compute_time_features(current_datetime):
    latitude = 52
    day_of_year = current_datetime.timetuple().tm_yday
    declination = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
    lat_rad = np.deg2rad(latitude)
    decl_rad = np.deg2rad(declination)
    hour_angle = np.arccos(-np.tan(lat_rad) * np.tan(decl_rad))
    day_length = (2 * hour_angle) * 24 / (2 * np.pi) * 60   # minutes, to match the training data

    hour = current_datetime.hour
    month = current_datetime.month
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return day_length, hour_sin, hour_cos, month_sin, month_cos


# Predict Power[kW] for a city, horizon and UTC time
def get_forecast(city_name, horizon_choice, current_datetime):
    if horizon_choice not in allowed_horizons:
        print('Error: horizon not supported:', horizon_choice)
        return None

    weather = get_weather_at(city_name, current_datetime)
    if weather is None:
        return None

    if weather['is_day'] == 0 or weather['GHI'] <= 0.1:
        print("It's nighttime — no solar output expected")
        return 0.0

    day_length, hour_sin, hour_cos, month_sin, month_cos = compute_time_features(current_datetime)

    row = pd.DataFrame([{
        'GHI': weather['GHI'],
        'temp': weather['temp'],
        'pressure': weather['pressure'],
        'humidity': weather['humidity'],
        'wind_speed': weather['wind_speed'],
        'rain_1h': weather['rain_1h'],
        'snow_1h': weather['snow_1h'],
        'clouds_all': weather['clouds_all'],
        'dayLength': day_length,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'month_sin': month_sin,
        'month_cos': month_cos
    }])
    row = row[features_weather_only]

    model_path = os.path.join(MODELS_DIR, f'forecast_weather_only_{horizon_choice}.joblib')
    model = joblib.load(model_path)
    prediction = model.predict(row)
    return prediction[0]
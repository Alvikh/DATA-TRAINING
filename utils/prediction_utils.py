import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_model_components(model_path='models/energy_model.pkl',
                         scaler_path='models/scaler.pkl',
                         features_path='models/model_features.pkl'):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model atau komponennya: {e}")

def preprocess_input(input_data, scaler, selected_features):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    if 'measured_at' in df.columns:
        df['measured_at'] = pd.to_datetime(df['measured_at'])
        df['hour'] = df['measured_at'].dt.hour
        df['day_of_week'] = df['measured_at'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['volt_curr'] = df['voltage'] * df['current']
    df['curr_squared'] = df['current'] ** 2
    df['temp_humid'] = df['temperature'] * df['humidity']

    df = df[selected_features].copy()
    df[selected_features] = scaler.transform(df[selected_features])

    return df

def predict_energy_kwh(input_data, duration_minutes=5):
    try:
        model, scaler, features = load_model_components()
        processed = preprocess_input(input_data, scaler, features)
        pred_power = model.predict(processed)[0]
        duration_hours = duration_minutes / 60.0
        pred_kwh = (pred_power * duration_hours) / 1000.0
        return float(pred_kwh)
    except Exception as e:
        raise ValueError(f"Gagal melakukan prediksi energi kWh: {e}")

def generate_future_dates(start_date, duration_type, num_periods=1):
    future_dates = []
    current_date = start_date

    if duration_type == 'day':
        for _ in range(num_periods * 24):
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    elif duration_type == 'week':
        for _ in range(num_periods * 7 * 24):
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    elif duration_type == 'month':
        for _ in range(num_periods * 30 * 24):
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    elif duration_type == 'year':
        for _ in range(num_periods * 365 * 24):
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    else:
        raise ValueError("duration_type harus 'day', 'week', 'month', atau 'year'")

    return future_dates

def predict_future_energy(start_data, duration_type='day', num_periods=1):
    try:
        model, scaler, features = load_model_components()
        start_date = datetime.now() if 'measured_at' not in start_data else pd.to_datetime(start_data['measured_at'])
        
        future_dates = generate_future_dates(start_date, duration_type, num_periods)
        results = []
        
        for date in future_dates:
            data_point = start_data.copy()
            data_point['measured_at'] = date
            
            processed = preprocess_input(data_point, scaler, features)
            pred_power = model.predict(processed)[0]
            pred_kwh = (pred_power * 1) / 1000.0  # 1 jam duration
            
            results.append({
                'timestamp': date,
                'predicted_power_watt': pred_power,
                'predicted_energy_kwh': pred_kwh
            })
            
        return pd.DataFrame(results)
    
    except Exception as e:
        raise ValueError(f"Gagal melakukan prediksi masa depan: {e}")

def save_prediction_plot(df, title="Prediksi Penggunaan Daya"):
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(df['timestamp'], df['predicted_power_watt'], 
                marker='o', linestyle='-', markersize=3, color='skyblue')
        plt.title(title)
        plt.xlabel("Waktu")
        plt.ylabel("Daya Prediksi (Watt)")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, f"prediction_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        raise ValueError(f"Gagal menyimpan plot prediksi: {e}")
def prepare_future_data(future_dates, last_sensor_data, selected_features):
    """
    Mempersiapkan DataFrame untuk prediksi masa depan.
    
    Args:
        future_dates (list): Daftar tanggal untuk prediksi
        last_sensor_data (dict): Data sensor terakhir
        selected_features (list): Daftar fitur yang digunakan model
        
    Returns:
        pd.DataFrame: DataFrame yang siap untuk prediksi
    """
    future_data_list = []
    
    for dt in future_dates:
        row = {
            'measured_at': dt,
            'voltage': last_sensor_data.get('voltage', 220),
            'current': last_sensor_data.get('current', 1.5),
            'temperature': last_sensor_data.get('temperature', 25),
            'humidity': last_sensor_data.get('humidity', 60),
            'energy': last_sensor_data.get('energy', 0),
            'frequency': last_sensor_data.get('frequency', 50),
            'power_factor': last_sensor_data.get('power_factor', 0.9)
        }
        future_data_list.append(row)
    
    future_df = pd.DataFrame(future_data_list)
    return future_df
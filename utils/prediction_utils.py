import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import os

def load_model_and_scaler(model_path, scaler_path):
    """Memuat model dan scaler yang sudah disimpan"""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def generate_future_dates(start_date, days=7):
    """Membuat rangkaian tanggal untuk prediksi"""
    return [start_date + timedelta(days=i) for i in range(days)]

def prepare_future_data(base_values, future_dates):
    """Mempersiapkan data untuk prediksi masa depan"""
    future_data = []
    
    for date in future_dates:
        hour = 12
        day_of_week = date.weekday()
        month = date.month
        is_weekend = int(day_of_week >= 5)
        
        record = {
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            **base_values
        }
        future_data.append(record)
    
    return pd.DataFrame(future_data)

def predict_future(model, scaler, future_df):
    """Melakukan prediksi untuk data masa depan"""
    features = [
        'voltage', 'current', 'power', 'energy', 'frequency',
        'power_factor', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month', 'is_weekend'
    ]
    
    X = future_df[features].copy()
    
    numeric_features = [
        'voltage', 'current', 'power', 'energy', 
        'frequency', 'temperature', 'humidity'
    ]
    X[numeric_features] = scaler.transform(X[numeric_features])
    
    predictions = model.predict(X)
    future_df['predicted_output'] = predictions
    
    return future_df

def generate_plot(results_df):
    """Generate plot image and return as base64 string"""
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestamp'], results_df['predicted_output'], 
             marker='o', linestyle='-', color='b', label='Prediksi')
    plt.title('Prediksi Konsumsi Energi')
    plt.xlabel('Tanggal')
    plt.ylabel('Konsumsi Energi (kWh)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
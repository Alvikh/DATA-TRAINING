import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import io
import base64

app = Flask(__name__)
app.debug = True

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
        # Ekstrak fitur waktu
        hour = 12  # Asumsi prediksi untuk tengah hari
        day_of_week = date.weekday()
        month = date.month
        is_weekend = int(day_of_week >= 5)
        
        # Gabungkan dengan fitur dasar
        record = {
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            **base_values  # Nilai-nilai dasar (voltage, current, dll)
        }
        future_data.append(record)
    
    return pd.DataFrame(future_data)

def predict_future(model, scaler, future_df):
    """Melakukan prediksi untuk data masa depan"""
    # Fitur yang digunakan model
    features = [
        'voltage', 'current', 'power', 'energy', 'frequency',
        'power_factor', 'temperature', 'humidity',
        'hour', 'day_of_week', 'month', 'is_weekend'
    ]
    
    # Persiapan data
    X = future_df[features].copy()
    
    # Standardisasi fitur numerik
    numeric_features = [
        'voltage', 'current', 'power', 'energy', 
        'frequency', 'temperature', 'humidity'
    ]
    X[numeric_features] = scaler.transform(X[numeric_features])
    
    # Prediksi
    predictions = model.predict(X)
    future_df['predicted_output'] = predictions
    
    return future_df

def generate_plot(results_df):
    """Generate plot image and return as base64 string"""
    plt.figure(figsize=(12, 6))
    
    # Plot garis prediksi
    plt.plot(results_df['timestamp'], results_df['predicted_output'], 
             marker='o', linestyle='-', color='b', label='Prediksi')
    
    # Formatting plot
    plt.title('Prediksi Konsumsi Energi 7 Hari ke Depan')
    plt.xlabel('Tanggal')
    plt.ylabel('Konsumsi Energi (kWh)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    # Convert to base64 string
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return plot_data

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Konfigurasi model
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
    try:
        # Memuat model dan scaler
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        # Mendapatkan input dari request
        if request.method == 'POST':
            data = request.get_json()
        else:  # GET request dengan parameter query
            data = request.args.to_dict()
        
        # Default values
        base_input = {
            'voltage': float(data.get('voltage', 220)),
            'current': float(data.get('current', 5.2)),
            'power': float(data.get('power', 1100)),
            'energy': float(data.get('energy', 2.5)),
            'frequency': float(data.get('frequency', 50)),
            'power_factor': float(data.get('power_factor', 0.95)),
            'temperature': float(data.get('temperature', 28)),
            'humidity': float(data.get('humidity', 65))
        }
        
        days = int(data.get('days', 7))
        
        # Generate dates
        start_date = datetime.now() + timedelta(days=1)
        future_dates = generate_future_dates(start_date, days=days)
        
        # Prepare data and predict
        future_df = prepare_future_data(base_input, future_dates)
        results_df = predict_future(model, scaler, future_df)
        
        # Generate plot
        plot_data = generate_plot(results_df)
        
        # Format results
        predictions = results_df[['timestamp', 'predicted_output']].rename(
            columns={'timestamp': 'date', 'predicted_output': 'energy_consumption_kWh'}
        ).to_dict('records')
        
        # Return response
        response = {
            'status': 'success',
            'predictions': predictions,
            'plot_image': plot_data,
            'input_parameters': base_input,
            'prediction_days': days
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
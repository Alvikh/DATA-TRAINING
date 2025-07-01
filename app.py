import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
import io
import base64
import os

app = Flask(__name__)

# Konfigurasi path yang lebih robust
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'energy_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Middleware untuk logging
@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

# Endpoint untuk health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'server': 'Energy Prediction API'
    })

# Endpoint untuk info model
@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        # Cek apakah model ada
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found',
                'model_exists': os.path.exists(MODEL_PATH),
                'scaler_exists': os.path.exists(SCALER_PATH)
            }), 404
        
        # Dapatkan info terakhir modifikasi
        model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        scaler_mtime = datetime.fromtimestamp(os.path.getmtime(SCALER_PATH))
        
        return jsonify({
            'status': 'success',
            'model_last_modified': model_mtime.isoformat(),
            'scaler_last_modified': scaler_mtime.isoformat(),
            'model_size': os.path.getsize(MODEL_PATH),
            'scaler_size': os.path.getsize(SCALER_PATH)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint utama untuk prediksi
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Cek ketersediaan model
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found'
            }), 404
        
        # Memuat model dan scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Mendapatkan input dari request
        if request.method == 'POST':
            if request.content_type == 'application/json':
                data = request.get_json()
            else:
                data = request.form.to_dict()
        else:  # GET request
            data = request.args.to_dict()
        
        # Validasi input
        try:
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
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid parameter value: {str(e)}'
            }), 400
        
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
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'plot_image': plot_data,
            'input_parameters': base_input,
            'prediction_days': days,
            'prediction_date_range': {
                'start': future_dates[0].strftime('%Y-%m-%d'),
                'end': future_dates[-1].strftime('%Y-%m-%d')
            }
        })
    
    except Exception as e:
        app.logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'error_type': type(e).__name__
        }), 500

# Endpoint untuk download plot sebagai image
@app.route('/download-plot', methods=['GET'])
def download_plot():
    try:
        # Gunakan parameter yang sama dengan endpoint predict
        response = predict()
        
        if response.status_code != 200:
            return response
        
        data = response.get_json()
        plot_data = base64.b64decode(data['plot_image'])
        
        return send_file(
            io.BytesIO(plot_data),
            mimetype='image/png',
            as_attachment=True,
            download_name='energy_prediction.png'
        )
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Fungsi-fungsi helper (sama seperti sebelumnya)
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

if __name__ == "__main__":
    # Buat folder models jika belum ada
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
print(">>> 1")

from flask import Flask, request, jsonify, send_file
from datetime import datetime,timedelta
import os
import pandas as pd

import json
import logging

from utils.mqtt_handler import MQTTHandler
from utils.prediction_utils import (
    load_model_and_scaler,
    generate_future_dates, # Import fungsi baru
    prepare_future_data,   # Import fungsi baru
    predict_future,        # Import fungsi baru
    generate_plot
)
print(">>> 2")

app = Flask(__name__)

# Konfigurasi path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'energy_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
print(">>> 3")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print(">>> 4")

# Initialize MQTT Handler
mqtt_handler = MQTTHandler(
    broker='broker.hivemq.com',
    port=1883,
)
print(">>> 5")

# Connect to MQTT on startup
try:
    print(">>> Sebelum mqtt connect")
    mqtt_handler.connect()
    print(">>> Setelah mqtt connect")
except Exception as e:
    logger.error(f"Failed to initialize MQTT: {e}")

# Endpoint untuk health check
@app.route('/health', methods=['GET'])
def health_check():
    mqtt_status = "connected" if mqtt_handler.connected else "disconnected"
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'server': 'Energy Prediction API',
        'mqtt_status': mqtt_status
    })

@app.route('/mqtt-status', methods=['GET'])
def mqtt_status():
    """Diagnose MQTT connection status and error if any"""
    try:
        status = "connected" if mqtt_handler.connected else "disconnected"
        
        test_result = {}
        try:
            mqtt_handler.connect()  # attempt reconnect
            test_result['reconnect_attempt'] = 'success'
        except Exception as test_err:
            test_result['reconnect_attempt'] = f'failed: {str(test_err)}'
        
        return jsonify({
            'status': 'success',
            'mqtt_connection_status': status,
            'broker': mqtt_handler.broker,
            'port': mqtt_handler.port,
            'reconnect_test': test_result,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Endpoint untuk info model
@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found'
            }), 404
        
        model_mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
        scaler_mtime = datetime.fromtimestamp(os.path.getmtime(SCALER_PATH))
        
        return jsonify({
            'status': 'success',
            'model_last_modified': model_mtime.isoformat(),
            'scaler_last_modified': scaler_mtime.isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Endpoint untuk publish prediksi ke MQTT
@app.route('/publish-prediction', methods=['POST'])
def publish_prediction():
    try:
        # Get prediction data first
        # Asumsi ini memanggil api_predict() untuk single point prediction
        # Jika ingin publish future prediction, perlu endpoint terpisah
        prediction_response = api_predict() 
        if prediction_response.status_code != 200:
            return prediction_response
        
        data = prediction_response.get_json()
        
        # Publish to MQTT
        topic = f"{mqtt_handler.control_topic_prefix}/prediction"
        message = json.dumps({
            'predictions': data['predicted_energy'], # Sesuaikan jika api_predict mengembalikan format berbeda
            'input_parameters': {
                'device_id': data.get('device_id'),
                'measured_at': data.get('measured_at')
            }
        })
        
        if mqtt_handler.publish(topic, message):
            return jsonify({
                'status': 'success',
                'message': 'Prediction published to MQTT',
                'topic': topic
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to publish to MQTT'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found',
                'timestamp': datetime.now().isoformat()
            }), 404

        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON',
                'timestamp': datetime.now().isoformat()
            }), 400

        sensor_data = request.get_json()
        
        try:
            measured_time = datetime.strptime(sensor_data.get('measured_at', ''), '%d-%m-%Y %H:%M:%S')
        except (ValueError, TypeError):
            measured_time = datetime.now()

        # Daftar fitur numerik yang digunakan saat training (sesuai input Anda)
        numeric_features = [
            'voltage', 'current', 'energy', 
            'frequency', 'power_factor', 'temperature', 'humidity'
        ]
        
        # Daftar fitur waktu
        time_features = ['hour', 'day_of_week', 'month', 'is_weekend']

        # Gabungan semua fitur yang diharapkan model
        all_model_features = numeric_features + time_features

        # Validasi bahwa semua kunci yang dibutuhkan ada di sensor_data
        # Hanya cek untuk fitur numerik yang diharapkan dari input JSON
        required_input_numeric_keys = [
            'voltage', 'current', 'energy', 'frequency',
            'power_factor', 'temperature', 'humidity'
        ]
        for key in required_input_numeric_keys:
            if key not in sensor_data:
                raise KeyError(f"Missing required field: '{key}' in input JSON.")

        input_features = {
            'voltage': float(sensor_data['voltage']),
            'current': float(sensor_data['current']),
            'energy': float(sensor_data['energy']),
            'frequency': float(sensor_data['frequency']),
            'power_factor': float(sensor_data['power_factor']),
            'temperature': float(sensor_data['temperature']),
            'humidity': float(sensor_data['humidity']),
            'hour': measured_time.hour,
            'day_of_week': measured_time.weekday(),
            'month': measured_time.month,
            'is_weekend': 1 if measured_time.weekday() >= 5 else 0
        }

        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        # Buat DataFrame dengan urutan kolom yang benar
        input_df = pd.DataFrame([input_features], columns=all_model_features)
        
        # Scale fitur numerik
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        prediction = model.predict(input_df)[0]

        return jsonify({
            'status': 'success',
            'device_id': sensor_data.get('id', 'unknown'),
            'measured_at': measured_time.strftime('%d-%m-%Y %H:%M:%S'),
            'predicted_energy': float(prediction),
            'units': 'Watt', # Asumsi target 'power' adalah Watt
            'timestamp': datetime.now().isoformat()
        })

    except KeyError as e:
        return jsonify({
            'status': 'error',
            'message': f'Missing required field: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 400
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid data value: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# --- ENDPOINT BARU UNTUK PREDIKSI MASA DEPAN ---
@app.route('/api/predict-future', methods=['POST'])
def api_predict_future():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found',
                'timestamp': datetime.now().isoformat()
            }), 404

        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON',
                'timestamp': datetime.now().isoformat()
            }), 400

        request_data = request.get_json()
        logger.info(f"Received request data: {request_data}")

        # Validasi input durasi prediksi
        duration_type = request_data.get('duration_type')  # 'day', 'week', 'month', 'year'
        if duration_type not in ['day', 'week', 'month', 'year']:
            return jsonify({
                'status': 'error',
                'message': "Required field 'duration_type' must be 'day', 'week', 'month', or 'year'.",
                'timestamp': datetime.now().isoformat()
            }), 400

        num_periods = int(request_data.get('num_periods', 1))  # Default 1 periode

        # Ambil data sensor terakhir sebagai baseline
        last_sensor_data = request_data.get('last_sensor_data')
        if not last_sensor_data:
            return jsonify({
                'status': 'error',
                'message': "Required field 'last_sensor_data' is missing. Provide baseline sensor values.",
                'timestamp': datetime.now().isoformat()
            }), 400

        required_baseline_keys = [
            'voltage', 'current', 'energy', 'frequency',
            'power_factor', 'temperature', 'humidity'
        ]
        for key in required_baseline_keys:
            if key not in last_sensor_data:
                raise KeyError(f"Missing required field in 'last_sensor_data': '{key}'.")
            try:
                last_sensor_data[key] = float(last_sensor_data[key])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid data type for '{key}' in 'last_sensor_data'. Must be numeric.")

        # Tentukan tanggal mulai prediksi
        start_date_str = request_data.get('start_date')  # Format 'DD-MM-YYYY HH:MM:SS'
        try:
            start_date = datetime.strptime(start_date_str, '%d-%m-%Y %H:%M:%S') if start_date_str else datetime.now()
        except (ValueError, TypeError):
            start_date = datetime.now()  # fallback

        # === Gunakan model dengan polynomial features ===
        model, scaler, features, poly_transformer = load_model_components()

        # Daftar fitur numerik dan waktu
        numeric_features = [
            'voltage', 'current', 'energy',
            'frequency', 'power_factor', 'temperature', 'humidity'
        ]
        time_features = ['hour', 'day_of_week', 'month', 'is_weekend']

        # 1. Generate future timestamps
        future_dates = generate_future_dates(start_date, duration_type, num_periods)

        # 2. Buat future_df dengan baseline sensor
        future_df = prepare_future_data(future_dates, last_sensor_data, numeric_features, time_features)

        # 3. Preprocess future data agar sesuai input saat training
        processed_df = preprocess_input(future_df, poly_transformer, scaler, features)

        # 4. Prediksi
        predictions = model.predict(processed_df)

        # 5. Format hasil prediksi
        formatted_predictions = []
        for i, pred in enumerate(predictions):
            formatted_predictions.append({
                'timestamp': future_dates[i].strftime('%d-%m-%Y %H:%M:%S'),
                'predicted_power': float(pred)  # Asumsi unit: Watt
            })

        # 6. Generate plot
        plot_path = generate_plot(future_dates, predictions, title=f"Prediksi Daya untuk {num_periods} {duration_type.capitalize()}")
        plot_url = f"/plots/{os.path.basename(plot_path)}"

        return jsonify({
            'status': 'success',
            'duration_type': duration_type,
            'num_periods': num_periods,
            'start_date': start_date.strftime('%d-%m-%Y %H:%M:%S'),
            'predictions': formatted_predictions,
            'plot_url': plot_url,
            'timestamp': datetime.now().isoformat()
        })

    except KeyError as e:
        return jsonify({
            'status': 'error',
            'message': f'Missing required field: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 400
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid data value: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 400
    except Exception as e:
        logger.error(f"Future prediction error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


# Endpoint untuk melayani file plot statis
@app.route('/plots/<filename>')
def serve_plot(filename):
    plot_dir = os.path.join(BASE_DIR, 'plots')
    return send_file(os.path.join(plot_dir, filename), mimetype='image/png')


@app.route('/mqtt-data', methods=['GET'])
def get_mqtt_data():
    """Endpoint to get latest MQTT data"""
    try:
        topic_filter = request.args.get('topic')
        data = mqtt_handler.get_latest_data(topic_filter)
        
        if topic_filter and not data:
            return jsonify({
                'status': 'error',
                'message': f'No data found for topic: {topic_filter}',
                'timestamp': datetime.now().isoformat()
            }), 404
            
        return jsonify({
            'status': 'success',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in /mqtt-data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/mqtt-data/<path:topic>', methods=['GET'])
def get_mqtt_data_by_topic(topic):
    """Endpoint untuk mendapatkan data terakhir dari topic tertentu"""
    try:
        data = mqtt_handler.get_latest_data(topic)
        if not data:
            return jsonify({
                'status': 'error',
                'message': f'No data found for topic: {topic}'
            }), 404
        
        return jsonify({
            'status': 'success',
            'topic': topic,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

      
if __name__ == "__main__":
    # Buat folder models dan plots jika belum ada
    print(">>> MASUK __main__ <<<")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    plot_dir = os.path.join(BASE_DIR, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Connect to MQTT
    try:
        mqtt_handler.connect()
    except Exception as e:
        logger.error(f"Failed to initialize MQTT: {e}")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
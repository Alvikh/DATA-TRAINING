from flask import Flask, request, jsonify, send_file
from datetime import datetime
import os

import json
import logging

from utils.mqtt_handler import MQTTHandler
from utils.prediction_utils import (
    load_model_and_scaler,
    generate_future_dates,
    prepare_future_data,
    predict_future,
    generate_plot
)
# from utils import message_handler as MessageHandler
app = Flask(__name__)

# Konfigurasi path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'energy_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MQTT Handler
mqtt_handler = MQTTHandler(
    broker='broker.hivemq.com',
    port=1883,
)

# Connect to MQTT on startup
try:
    mqtt_handler.connect()
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
        # Status koneksi
        status = "connected" if mqtt_handler.connected else "disconnected"
        
        # Tes koneksi langsung ke broker (jika perlu bisa lebih kompleks)
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
        prediction_response = predict()
        if prediction_response.status_code != 200:
            return prediction_response
        
        data = prediction_response.get_json()
        
        # Publish to MQTT
        topic = f"{mqtt_handler.control_topic_prefix}/prediction"
        message = json.dumps({
            'predictions': data['predictions'],
            'input_parameters': data['input_parameters']
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

# Endpoint utama untuk prediksi (sama seperti sebelumnya)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found'
            }), 404
        
        model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
        
        if request.method == 'POST':
            if request.content_type == 'application/json':
                data = request.get_json()
            else:
                data = request.form.to_dict()
        else:
            data = request.args.to_dict()
        
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
        
        start_date = datetime.now() + timedelta(days=1)
        future_dates = generate_future_dates(start_date, days=days)
        future_df = prepare_future_data(base_input, future_dates)
        results_df = predict_future(model, scaler, future_df)
        plot_data = generate_plot(results_df)
        
        predictions = results_df[['timestamp', 'predicted_output']].rename(
            columns={'timestamp': 'date', 'predicted_output': 'energy_consumption_kWh'}
        ).to_dict('records')
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'plot_image': plot_data,
            'input_parameters': base_input,
            'prediction_days': days
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
@app.route('/mqtt-data', methods=['GET'])
def get_mqtt_data():
    """Endpoint to get latest MQTT data"""
    try:
        # Get topic filter from query parameter if exists
        topic_filter = request.args.get('topic')
        
        # Get data from MQTT handler
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
    # Buat folder models jika belum ada
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Connect to MQTT
    try:
        mqtt_handler.connect()
    except Exception as e:
        logger.error(f"Failed to initialize MQTT: {e}")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
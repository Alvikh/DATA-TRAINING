import joblib
import pandas as pd
from datetime import datetime

def predict_new_data(model_path, scaler_path, input_data):
    """Membuat prediksi dari input baru"""
    try:
        # Memuat model dan scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Konversi input ke DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ekstraksi fitur waktu dari timestamp
        timestamp = datetime.strptime(input_data['timestamp'], '%Y-%m-%d %H:%M:%S')
        input_df['hour'] = timestamp.hour
        input_df['day_of_week'] = timestamp.weekday()
        input_df['month'] = timestamp.month
        input_df['is_weekend'] = int(timestamp.weekday() >= 5)
        
        # Hapus kolom timestamp yang tidak digunakan model
        input_df.drop('timestamp', axis=1, inplace=True, errors='ignore')
        
        # Pastikan urutan kolom sama seperti saat training
        expected_features = [
            'voltage', 'current', 'power', 'energy', 'frequency',
            'power_factor', 'temperature', 'humidity',
            'hour', 'day_of_week', 'month', 'is_weekend'
        ]
        
        # Reorder kolom sesuai ekspektasi model
        input_df = input_df[expected_features]
        
        # Standardisasi fitur numerik
        numeric_features = ['voltage', 'current', 'power', 'energy', 
                          'frequency', 'temperature', 'humidity']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        # Prediksi
        prediction = model.predict(input_df)
        
        return prediction[0]
    
    except Exception as e:
        print(f"Error dalam prediksi: {str(e)}")
        return None

if __name__ == "__main__":
    # Contoh input baru (pastikan format sama dengan data training)
    new_input = {
        'timestamp': '2023-07-15 14:30:00',  # Format: YYYY-MM-DD HH:MM:SS
        'voltage': 220,
        'current': 5.2,
        'power': 1100,
        'energy': 2.5,
        'frequency': 50,
        'power_factor': 0.95,
        'temperature': 28,
        'humidity': 65
    }
    
    # Path model
    MODEL_PATH = "models/energy_model.pkl"
    SCALER_PATH = "models/scaler.pkl"
    
    # Prediksi
    pred = predict_new_data(MODEL_PATH, SCALER_PATH, new_input)
    if pred is not None:
        print(f"\nPrediksi output_energy: {pred:.2f}")
    else:
        print("Gagal melakukan prediksi. Periksa error di atas.")
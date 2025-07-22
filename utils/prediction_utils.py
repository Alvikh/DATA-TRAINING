# # utils/prediction_utils.py

# import os
# from datetime import datetime, timedelta

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# # === FUNGSI LAMA ===

# def load_model_and_scaler(model_path, scaler_path):
#     """Memuat model dan scaler yang sudah terlatih."""
#     try:
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         return model, scaler
#     except Exception as e:
#         raise Exception(f"Gagal memuat model atau scaler: {e}")

# # === FUNGSI UNTUK PREDIKSI MASA DEPAN ===

# def generate_future_dates(start_date, duration_type, num_periods=1):
#     """
#     Menghasilkan daftar tanggal di masa depan berdasarkan durasi.
#     Args:
#         start_date (datetime): Tanggal mulai prediksi.
#         duration_type (str): 'day', 'week', 'month', 'year'.
#         num_periods (int): Berapa banyak periode ke depan.
#     Returns:
#         list: Daftar objek datetime untuk periode prediksi.
#     """
#     future_dates = []
#     current_date = start_date

#     if duration_type == 'day':
#         for _ in range(num_periods * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'week':
#         for _ in range(num_periods * 7 * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'month':
#         for _ in range(num_periods * 30 * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'year':
#         for _ in range(num_periods * 365 * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     else:
#         raise ValueError("duration_type tidak valid. Gunakan 'day', 'week', 'month', atau 'year'.")

#     return future_dates


# def prepare_future_data(future_dates, last_sensor_data, numeric_features, time_features):
#     """
#     Mempersiapkan DataFrame untuk prediksi masa depan.
#     Menggunakan data sensor terakhir sebagai baseline untuk fitur non-waktu.
#     Args:
#         future_dates (list): Daftar objek datetime untuk prediksi.
#         last_sensor_data (dict): Data sensor terakhir (voltage, current, energy, frequency, power_factor, temperature, humidity).
#         numeric_features (list): Daftar nama fitur numerik yang akan diskalakan.
#         time_features (list): Daftar nama fitur waktu.
#     Returns:
#         pd.DataFrame: DataFrame yang siap untuk prediksi.
#     """
#     future_data_list = []
#     for dt in future_dates:
#         row = {
#             'voltage': last_sensor_data['voltage'],
#             'current': last_sensor_data['current'],
#             'energy': last_sensor_data['energy'],
#             'frequency': last_sensor_data['frequency'],
#             'power_factor': last_sensor_data['power_factor'],
#             'temperature': last_sensor_data['temperature'],
#             'humidity': last_sensor_data['humidity'],
#             'hour': dt.hour,
#             'day_of_week': dt.weekday(),
#             'month': dt.month,
#             'is_weekend': 1 if dt.weekday() >= 5 else 0
#         }
#         future_data_list.append(row)

#     future_df = pd.DataFrame(future_data_list)
#     all_model_features = numeric_features + time_features
#     future_df = future_df[all_model_features]
#     return future_df


# def predict_future(model, scaler, future_df, numeric_features):
#     """
#     Melakukan prediksi untuk data masa depan.
#     Args:
#         model: Model machine learning yang sudah terlatih.
#         scaler: Scaler yang sudah terlatih.
#         future_df (pd.DataFrame): DataFrame berisi fitur untuk prediksi masa depan.
#         numeric_features (list): Daftar nama fitur numerik yang akan diskalakan.
#     Returns:
#         np.array: Array prediksi.
#     """
#     df_scaled = future_df.copy()
#     df_scaled[numeric_features] = scaler.transform(future_df[numeric_features])
#     predictions = model.predict(df_scaled)
#     return predictions


# def generate_plot(dates, predictions, title="Prediksi Penggunaan Daya"):
#     """
#     Menghasilkan plot prediksi dan menyimpannya sebagai gambar.
#     Args:
#         dates (list): Daftar objek datetime.
#         predictions (np.array): Array prediksi daya.
#         title (str): Judul plot.
#     Returns:
#         str: Path ke file gambar yang disimpan.
#     """
#     plt.figure(figsize=(15, 7))
#     plt.plot(dates, predictions, marker='o', linestyle='-', markersize=3, color='skyblue')
#     plt.title(title)
#     plt.xlabel("Waktu")
#     plt.ylabel("Daya Prediksi (Watt)")
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()

#     plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
#     if not os.path.exists(plot_dir):
#         os.makedirs(plot_dir)

#     plot_path = os.path.join(plot_dir, f"prediction_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
#     plt.savefig(plot_path)
#     plt.close()
#     return plot_path


# # === FUNGSI BARU UNTUK MODEL DENGAN FITUR INTERAKSI DAN POLYNOMIAL ===
# def load_model_components(model_path='models/energy_model.pkl',
#                           scaler_path='models/scaler.pkl',
#                           features_path='models/model_features.pkl',
#                           poly_path='models/poly_transformer.pkl'):
#     try:
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         features = joblib.load(features_path)
#         poly_transformer = joblib.load(poly_path) if os.path.exists(poly_path) else None
#         return model, scaler, features, poly_transformer
#     except Exception as e:
#         raise RuntimeError(f"Gagal memuat model atau komponennya: {e}")



# def preprocess_input(input_data, poly_transformer=None, scaler=None, selected_features=None):
#     if isinstance(input_data, dict):
#         df = pd.DataFrame([input_data])
#     else:
#         df = input_data.copy()

#     if 'measured_at' in df.columns:
#         df['measured_at'] = pd.to_datetime(df['measured_at'])
#         df['hour'] = df['measured_at'].dt.hour
#         df['day_of_week'] = df['measured_at'].dt.dayofweek
#         df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
#     if 'hour' in df.columns and 'sin_hour' not in df.columns:
#         df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
#         df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

#     df['volt_curr'] = df['voltage'] * df['current']
#     df['curr_squared'] = df['current'] ** 2
#     df['temp_humid'] = df['temperature'] * df['humidity']

#     base_columns = ['voltage', 'current', 'energy', 'frequency', 'power_factor',
#                     'temperature', 'humidity', 'volt_curr', 'curr_squared', 'temp_humid',
#                     'sin_hour', 'cos_hour', 'is_weekend']

#     df_features = df[base_columns].copy()   

#     if poly_transformer:
#         df_poly = poly_transformer.transform(df_features)
#         df_poly = pd.DataFrame(df_poly, columns=poly_transformer.get_feature_names_out(df_features.columns))
#         df_features = df_poly

#     if scaler and selected_features:
#         df_features.loc[:, selected_features] = scaler.transform(df_features[selected_features])

#     return df_features[selected_features]


# def predict_energy_kwh(input_data, duration_minutes=5):
#     model, scaler, features, poly_transformer = load_model_components()

#     try:
#         processed = preprocess_input(input_data, poly_transformer, scaler, features)
#         pred_log = model.predict(processed)
#         pred_power = np.expm1(pred_log)  # hasil prediksi dalam Watt
#         duration_hours = duration_minutes / 60.0
#         pred_kwh = (pred_power[0] * duration_hours) / 1000
#         return float(pred_kwh)
#     except Exception as e:
#         raise ValueError(f"Gagal melakukan prediksi energi kWh: {e}")
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
def prepare_future_data(future_dates, last_sensor_data, numeric_features, time_features):
    """
    Mempersiapkan DataFrame untuk prediksi masa depan.
    Menggunakan data sensor terakhir sebagai baseline untuk fitur non-waktu.

    Args:
        future_dates (list): Daftar objek datetime untuk prediksi.
        last_sensor_data (dict): Data sensor terakhir (voltage, current, energy, frequency, power_factor, temperature, humidity).
        numeric_features (list): Daftar nama fitur numerik yang akan diskalakan.
        time_features (list): Daftar nama fitur waktu.

    Returns:
        pd.DataFrame: DataFrame yang siap untuk prediksi.
    """
    future_data_list = []
    for dt in future_dates:
        row = {
            'voltage': float(last_sensor_data.get('voltage', 220)),
            'current': float(last_sensor_data.get('current', 1.5)),
            'energy': float(last_sensor_data.get('energy', 0)),
            'frequency': float(last_sensor_data.get('frequency', 50)),
            'power_factor': float(last_sensor_data.get('power_factor', 0.9)),
            'temperature': float(last_sensor_data.get('temperature', 25)),
            'humidity': float(last_sensor_data.get('humidity', 60)),
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
        future_data_list.append(row)

    future_df = pd.DataFrame(future_data_list)

    # Pastikan hanya kolom-kolom yang digunakan oleh model yang disertakan
    all_model_features = numeric_features + time_features
    future_df = future_df[all_model_features]

    return future_df

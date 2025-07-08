# # utils/prediction_utils.py
# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# # Fungsi yang sudah ada (load_model_and_scaler)
# def load_model_and_scaler(model_path, scaler_path):
#     """Memuat model dan scaler yang sudah terlatih."""
#     try:
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         return model, scaler
#     except Exception as e:
#         raise Exception(f"Gagal memuat model atau scaler: {e}")

# # --- FUNGSI BARU UNTUK PREDIKSI MASA DEPAN ---

# def generate_future_dates(start_date, duration_type, num_periods=1):
#     """
#     Menghasilkan daftar tanggal di masa depan berdasarkan durasi.
#     Args:
#         start_date (datetime): Tanggal mulai prediksi.
#         duration_type (str): 'week', 'month', 'year'.
#         num_periods (int): Berapa banyak periode (minggu/bulan/tahun) ke depan.
#     Returns:
#         list: Daftar objek datetime untuk periode prediksi.
#     """
#     future_dates = []
#     current_date = start_date

#     if duration_type == 'day':
#         # Prediksi per jam untuk 24 jam
#         for _ in range(num_periods * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'week':
#         # Prediksi per jam untuk 7 hari (168 jam)
#         for _ in range(num_periods * 7 * 24):
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'month':
#         # Prediksi per jam untuk 30 hari (720 jam)
#         for _ in range(num_periods * 30 * 24):  # Asumsi 30 hari per bulan
#             future_dates.append(current_date)
#             current_date += timedelta(hours=1)
#     elif duration_type == 'year':
#         # Prediksi per jam untuk 365 hari (8760 jam)
#         for _ in range(num_periods * 365 * 24):  # Asumsi 365 hari per tahun
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
#     # Buat list dictionaries untuk setiap baris data masa depan
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

#     # Pastikan urutan kolom sesuai dengan yang diharapkan model saat pelatihan
#     # Ini adalah daftar lengkap fitur yang diharapkan model (numerik + waktu)
#     # Sesuaikan ini jika model Anda dilatih dengan urutan yang berbeda!
#     all_model_features = numeric_features + time_features
#     future_df = future_df[all_model_features] # Reorder columns

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
#     # Pastikan hanya fitur numerik yang diskalakan
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

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load model, scaler, features, dan poly_transformer
def load_model_components(model_path='models/energy_model.pkl',
                          scaler_path='models/scaler.pkl',
                          features_path='models/model_features.pkl',
                          poly_path='models/poly_transformer.pkl'):
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        poly_transformer = joblib.load(poly_path) if os.path.exists(poly_path) else None
        return model, scaler, features, poly_transformer
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model atau komponennya: {e}")

# Persiapan fitur dari input (dalam format dictionary atau DataFrame)
def preprocess_input(input_data, poly_transformer=None, scaler=None, selected_features=None):
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()

    # Fitur waktu
    if 'measured_at' in df.columns:
        df['measured_at'] = pd.to_datetime(df['measured_at'])
        df['hour'] = df['measured_at'].dt.hour
        df['day_of_week'] = df['measured_at'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Fitur interaksi
    df['volt_curr'] = df['voltage'] * df['current']
    df['curr_squared'] = df['current'] ** 2
    df['temp_humid'] = df['temperature'] * df['humidity']

    base_columns = ['voltage', 'current', 'energy', 'frequency', 'power_factor',
                    'temperature', 'humidity', 'volt_curr', 'curr_squared', 'temp_humid',
                    'sin_hour', 'cos_hour', 'is_weekend']

    df_features = df[base_columns]

    # Polynomial transform
    if poly_transformer:
        df_poly = poly_transformer.transform(df_features)
        df_poly = pd.DataFrame(df_poly, columns=poly_transformer.get_feature_names_out(df_features.columns))
        df_features = df_poly

    # Scaling
    if scaler and selected_features:
        df_features[selected_features] = scaler.transform(df_features[selected_features])

    return df_features[selected_features]

# Fungsi prediksi
def predict_energy_kwh(input_data, duration_minutes=5):
    model, scaler, features, poly_transformer = load_model_components()

    try:
        processed = preprocess_input(input_data, poly_transformer, scaler, features)
        pred_log = model.predict(processed)
        pred_power = np.expm1(pred_log)  # hasil prediksi dalam Watt
        duration_hours = duration_minutes / 60.0
        pred_kwh = (pred_power[0] * duration_hours) / 1000
        return float(pred_kwh)
    except Exception as e:
        raise ValueError(f"Gagal melakukan prediksi energi kWh: {e}")



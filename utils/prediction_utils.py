# utils/prediction_utils.py
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Fungsi yang sudah ada (load_model_and_scaler)
def load_model_and_scaler(model_path, scaler_path):
    """Memuat model dan scaler yang sudah terlatih."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise Exception(f"Gagal memuat model atau scaler: {e}")

# --- FUNGSI BARU UNTUK PREDIKSI MASA DEPAN ---

def generate_future_dates(start_date, duration_type, num_periods=1):
    """
    Menghasilkan daftar tanggal di masa depan berdasarkan durasi.
    Args:
        start_date (datetime): Tanggal mulai prediksi.
        duration_type (str): 'week', 'month', 'year'.
        num_periods (int): Berapa banyak periode (minggu/bulan/tahun) ke depan.
    Returns:
        list: Daftar objek datetime untuk periode prediksi.
    """
    future_dates = []
    current_date = start_date

    if duration_type == 'week':
        # Prediksi per jam untuk 7 hari (168 jam)
        for _ in range(num_periods * 7 * 24):
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    elif duration_type == 'month':
        # Prediksi per jam untuk 30 hari (720 jam)
        for _ in range(num_periods * 30 * 24): # Asumsi 30 hari per bulan
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    elif duration_type == 'year':
        # Prediksi per jam untuk 365 hari (8760 jam)
        for _ in range(num_periods * 365 * 24): # Asumsi 365 hari per tahun
            future_dates.append(current_date)
            current_date += timedelta(hours=1)
    else:
        raise ValueError("duration_type tidak valid. Gunakan 'week', 'month', atau 'year'.")
    
    return future_dates

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
    # Buat list dictionaries untuk setiap baris data masa depan
    future_data_list = []
    for dt in future_dates:
        row = {
            'voltage': last_sensor_data['voltage'],
            'current': last_sensor_data['current'],
            'energy': last_sensor_data['energy'],
            'frequency': last_sensor_data['frequency'],
            'power_factor': last_sensor_data['power_factor'],
            'temperature': last_sensor_data['temperature'],
            'humidity': last_sensor_data['humidity'],
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'is_weekend': 1 if dt.weekday() >= 5 else 0
        }
        future_data_list.append(row)
    
    future_df = pd.DataFrame(future_data_list)

    # Pastikan urutan kolom sesuai dengan yang diharapkan model saat pelatihan
    # Ini adalah daftar lengkap fitur yang diharapkan model (numerik + waktu)
    # Sesuaikan ini jika model Anda dilatih dengan urutan yang berbeda!
    all_model_features = numeric_features + time_features
    future_df = future_df[all_model_features] # Reorder columns

    return future_df

def predict_future(model, scaler, future_df, numeric_features):
    """
    Melakukan prediksi untuk data masa depan.
    Args:
        model: Model machine learning yang sudah terlatih.
        scaler: Scaler yang sudah terlatih.
        future_df (pd.DataFrame): DataFrame berisi fitur untuk prediksi masa depan.
        numeric_features (list): Daftar nama fitur numerik yang akan diskalakan.
    Returns:
        np.array: Array prediksi.
    """
    # Pastikan hanya fitur numerik yang diskalakan
    df_scaled = future_df.copy()
    df_scaled[numeric_features] = scaler.transform(future_df[numeric_features])
    
    predictions = model.predict(df_scaled)
    return predictions

def generate_plot(dates, predictions, title="Prediksi Penggunaan Daya"):
    """
    Menghasilkan plot prediksi dan menyimpannya sebagai gambar.
    Args:
        dates (list): Daftar objek datetime.
        predictions (np.array): Array prediksi daya.
        title (str): Judul plot.
    Returns:
        str: Path ke file gambar yang disimpan.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(dates, predictions, marker='o', linestyle='-', markersize=3, color='skyblue')
    plt.title(title)
    plt.xlabel("Waktu")
    plt.ylabel("Daya Prediksi (Watt)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plot_path = os.path.join(plot_dir, f"prediction_plot_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path